import math

import paddle
import paddle.nn as nn
from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils
from paddleseg.models import layers
from .ocrnet_nv import OCRNetNV


@manager.MODELS.add_component
class MscaleOCRNet(nn.Layer):
    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices,
                 n_scales=[0.5, 1.0, 2.0],
                 ocr_mid_channels=512,
                 ocr_key_channels=256,
                 align_corners=False,
                 pretrained=None):
        super().__init__()
        self.ocrnet = OCRNetNV(
            num_classes,
            backbone,
            backbone_indices,
            ocr_mid_channels=ocr_mid_channels,
            ocr_key_channels=ocr_key_channels,
            align_corners=align_corners,
            ms_attention=True)
        self.scale_attn = AttenHead(in_ch=ocr_mid_channels, out_ch=1)
        self.low2high_attn = Low2HighAttenHead(
            in_ch=2, out_ch=1, align_corners=align_corners)

        self.n_scales = n_scales
        self.pretrained = pretrained
        self.align_corners = align_corners

        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)

    def forward(self, x):
        if self.training:
            return self.two_scale_forward(x)
        else:
            return self.nscale_forward(x, self.n_scales)

    def two_scale_forward(self, x_1x):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        x_lo = nn.functional.interpolate(
            x_1x,
            scale_factor=0.5,
            align_corners=self.align_corners,
            mode='bilinear')
        lo_outs = self.single_scale_forward(x_lo)
        pred_05x = lo_outs['cls_out']
        p_lo = pred_05x
        aux_lo = lo_outs['aux_out']
        attn_lo = lo_outs['attn']
        attn_lo_ = attn_lo.detach()

        hi_outs = self.single_scale_forward(x_1x)
        pred_10x = hi_outs['cls_out']
        p_hi = pred_10x
        aux_hi = hi_outs['aux_out']
        attn_hi = hi_outs['attn']
        p_hi_ = p_hi.detach()
        aux_hi_ = aux_hi.detach()
        attn_hi_ = attn_hi.detach()

        # combine lo and hi predictions with attention
        p_lo = scale_as(p_lo * attn_lo, p_hi)
        joint_pred = p_lo + p_hi * attn_hi
        aux_lo = scale_as(aux_lo * attn_lo, p_hi)
        joint_aux = aux_lo + aux_hi * attn_hi

        new_attn_hi = self.low2high_attn(attn_lo_, attn_hi_)
        p_lo_ = p_lo.detach()
        joint_pred_ = p_lo_ + p_hi_ * new_attn_hi
        aux_lo_ = aux_lo.detach()
        joint_aux_ = aux_lo_ + aux_hi_ * new_attn_hi

        # Optionally, apply supervision to the multi-scale predictions
        # directly.
        scaled_pred_05x = scale_as(pred_05x, p_hi)
        output = [joint_pred, joint_aux, scaled_pred_05x, pred_10x]
        output.extend(output)
        output.extend([joint_pred_, joint_aux_, joint_pred_, joint_aux_])
        return output

    def nscale_forward(self, x_1x, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        x_1x:
          scales - a list of scales to evaluate
          x_1x - dict containing 'images', the x_1x, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        #         # Lower resolution provides attention for higher rez predictions,
        #         # so we evaluate in order: high to low
        #         scales = sorted(scales, reverse=True)
        scales = sorted(scales)

        pred = None
        aux = None
        attn_outs = []

        for s in scales:
            x = nn.functional.interpolate(
                x_1x,
                scale_factor=s,
                align_corners=self.align_corners,
                mode='bilinear')
            outs = self.single_scale_forward(x)

            cls_out = outs['cls_out']
            attn_out = outs['attn']
            aux_out = outs['aux_out']

            if pred is None:
                pred = cls_out * attn_out
            elif s > 1.0:
                # downscale previous
                new_attn_out = self.low2high_attn(attn_outs[-1], attn_out)
                #                 print(attn_outs[-1].shape, attn_out.shape, new_attn_out.shape)
                cls_out = cls_out * new_attn_out
                pred += scale_as(cls_out, pred, self.align_corners)
            elif s == 1.0:
                pred = scale_as(pred, cls_out, self.align_corners)
                pred += cls_out * attn_out
            else:
                raise Exception()

            attn_outs.append(attn_out)
        return [pred]

    def single_scale_forward(self, x):
        x_size = x.shape[2:]
        cls_out, aux_out, ocr_mid_feats = self.ocrnet(x)
        attn = self.scale_attn(ocr_mid_feats)

        cls_out = nn.functional.interpolate(
            cls_out,
            size=x_size,
            mode='bilinear',
            align_corners=self.align_corners)
        aux_out = nn.functional.interpolate(
            aux_out,
            size=x_size,
            mode='bilinear',
            align_corners=self.align_corners)
        attn = nn.functional.interpolate(
            attn,
            size=x_size,
            mode='bilinear',
            align_corners=self.align_corners)

        return {'cls_out': cls_out, 'aux_out': aux_out, 'attn': attn}


class Low2HighAttenHead(nn.Layer):
    def __init__(self, in_ch, out_ch, align_corners):
        super(Low2HighAttenHead, self).__init__()
        self.align_corners = align_corners
        # bottleneck channels for seg and attn heads
        bot_ch = 32
        self.low2high_atten_head = nn.Sequential(
            layers.ConvBNReLU(in_ch, bot_ch, 3, padding=1, bias_attr=False),
            layers.ConvBNReLU(bot_ch, bot_ch, 3, padding=1, bias_attr=False),
            nn.Conv2D(bot_ch, out_ch, kernel_size=(1, 1), bias_attr=False),
            nn.Sigmoid())

    def forward(self, low_attn, high_attn):
        low_attn = scale_as(low_attn, high_attn, self.align_corners)
        fuse_attn = paddle.concat([low_attn, high_attn], axis=1)
        return self.low2high_atten_head(fuse_attn)


class AttenHead(nn.Layer):
    def __init__(self, in_ch, out_ch):
        super(AttenHead, self).__init__()
        # bottleneck channels for seg and attn heads
        bot_ch = 256

        self.atten_head = nn.Sequential(
            layers.ConvBNReLU(in_ch, bot_ch, 3, padding=1, bias_attr=False),
            layers.ConvBNReLU(bot_ch, bot_ch, 3, padding=1, bias_attr=False),
            nn.Conv2D(bot_ch, out_ch, kernel_size=(1, 1), bias_attr=False),
            nn.Sigmoid())

    def forward(self, x):
        return self.atten_head(x)


def scale_as(x, y, align_corners=False):
    '''
    scale x to the same size as y
    '''
    y_size = y.shape[2], y.shape[3]
    x_scaled = nn.functional.interpolate(
        x, size=y_size, mode='bilinear', align_corners=align_corners)
    return x_scaled

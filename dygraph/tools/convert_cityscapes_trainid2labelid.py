from paddleseg.datasets.cityscapes_labels import labels
from PIL import Image as PILImage
import numpy as np
import os


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def get_image_list(image_path):
    """Get image list"""
    valid_suffix = [
        '.JPEG', '.jpeg', '.JPG', '.jpg', '.BMP', '.bmp', '.PNG', '.png'
    ]
    image_list = []
    image_dir = None
    if os.path.isfile(image_path):
        if os.path.splitext(image_path)[-1] in valid_suffix:
            image_list.append(image_path)
    elif os.path.isdir(image_path):
        image_dir = image_path
        for root, dirs, files in os.walk(image_path):
            for f in files:
                if '.ipynb_checkpoints' in root:
                    continue
                if os.path.splitext(f)[-1] in valid_suffix:
                    image_list.append(os.path.join(root, f))
    else:
        raise FileNotFoundError(
            '`--image_path` is not found. it should be an image file or a directory including images'
        )

    if len(image_list) == 0:
        raise RuntimeError('There are not image file in `--image_path`')

    return image_list, image_dir


trainid2labelid = {label.trainId: label.id for label in reversed(labels)}

root_dir = '/home/chulutao/PaddleSeg/dygraph/saved_model/mscale_ocr_cityscapes_autolabel_whole_mapillary_boot_ce_weight_flip/test_predict_results/'
im_dir = root_dir + 'pseudo_color_prediction/'
result_dir = root_dir + 'convert_to_labelid/'
# image_list, _ = get_image_list(im_dir)
# print(len(image_list))
# for i, im_path in enumerate(image_list):
#     print('No.', i)
#     im_split = im_path.split('/')

#     im = PILImage.open(im_path)
#     im = np.asarray(im)
#     new_im = np.ones_like(im) * 200
#     print(np.unique(im))
#     for k in trainid2labelid:
#         new_im[im == k] = trainid2labelid[k]

#     print(np.unique(new_im))
#     new_im = PILImage.fromarray(new_im)
#     result_path = os.path.join(result_dir, im_split[-2], im_split[-1])
#     print(result_path)
#     mkdir(result_path)
#     new_im.save(result_path)

import shutil
for root, dirs, files in os.walk(result_dir):
    print(root)
    if '.ipynb_checkpoints' in root:
        print(root)
        shutil.rmtree(root)

print('after clear')
for root, dirs, files in os.walk(result_dir):
    if '.ipynb_checkpoints' in root:
        print(root)

import os
from pathlib import Path
import cv2


def split_img_mask(path):
    folder_names = os.listdir(path)
    images, masks = [], []
    for name in folder_names:
        img = cv2.imread(path + name)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        img = img[:, :, ::-1]
        images.append(img[:, :256])
        masks.append(img[:, 256:])  # possible have to reshape(256*256*3)
    return images, masks

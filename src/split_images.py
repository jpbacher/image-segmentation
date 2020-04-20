import os
import cv2
import tensorflow as tf


def split_to_arrays(dir_path):
    """
    Split the images into the actual image & its mask counterpart.
    :param dir_path: directory path of images
    :return: 2 lists of numpy arrays
    """
    file_names = os.listdir(dir_path)
    frames, masks = [], []
    for name in file_names:
        img = cv2.imread(dir_path + '/' + name)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        img = img[:, :, ::-1]
        frames.append(img[:, :256])
        masks.append(img[:, 256:])
    return frames, masks


def split_to_tensors(dir_path):
    """
    Split the images into 2 tensors, the actual & image and masks
    :param dir_path: directory path of images
    :return: 2 tensors
    """
    file_names = os.listdir(dir_path)
    img_path = [os.path.join(dir_path, f_name) for f_name in file_names]
    for img in img_path:
        img_str = tf.io.read_file(img)
        img_decode = tf.image.decode_jpeg(img_str)
        # ensure image is correct size
        out = tf.image.resize(img_decode, size=(256, 512))
        # keep the height and channels
        frames = out[:, :256, :]
        masks = out[:, 256:, :]
        return frames, masks


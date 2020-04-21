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


def get_tensor(filename):
    img_string = tf.io.read_file(filename)
    img_decode = tf.image.decode_jpeg(img_string)
    img = tf.image.convert_image_dtype(img_decode, tf.float32)
    img = tf.image.resize(img, size=(256, 512))
    return img


def get_frame(tensor, width=256):
    return tensor[:, :width, :]


def get_mask(tensor, width=256):
    return tensor[:, width:, :]

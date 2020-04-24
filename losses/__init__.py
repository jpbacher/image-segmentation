from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K


def dice_coeff(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersect + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_loss_bce(y_true, y_pred):
    bce = BinaryCrossentropy()
    dice_bce = 1 + bce(y_true, y_pred) - dice_coeff(y_true, y_pred)
    return dice_bce

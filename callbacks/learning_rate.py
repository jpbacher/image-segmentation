# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
# tfa.optimizers.CyclicalLearningRate


def lr_schedule(epoch):
    if epoch <= 12:
        return 1e-4
    elif 12 < epoch < 19:
        return 1e-6
    elif epoch < 21:
        return 1e-4
    elif epoch < 23:
        return 1e-5
    else:
        return 1e-6

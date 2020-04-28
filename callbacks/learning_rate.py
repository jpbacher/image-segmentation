from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend
# from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
# tfa.optimizers.CyclicalLearningRate


class FindLrRange(Callback):
    def __init__(self, start_lr, end_lr):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr

    def on_train_begin(self, logs={}):
        self.lrs = []
        self.losses = []
        tensorflow.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)
        n_steps = self.params['steps'] if self.params['steps'] is not None else round(
            self.params['samples'] / self.params['batch_size']
        )
        n_steps *= self.params['epochs']
        self.by = (self.end_lr - self.start_lr) / n_steps

    def on_batch_end(self, batch, logs={}):
        lr = float(tensorflow.keras.backend.get_value(self.model.optimizer.lr))
        self.lrs.append(lr)
        self.losses.append(logs.get('loss'))
        lr += self.by
        tensorflow.keras.backend.set_value(self.model.optimizer.lr, lr)


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

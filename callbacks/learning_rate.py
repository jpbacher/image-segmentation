import numpy as np
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import tensorflow.keras.backend as K


class FindLrRange(Callback):
    def __init__(self, start_lr, end_lr):
        super().__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr

    def on_train_begin(self, logs={}):
        self.lrs = []
        self.losses = []
        K.set_value(self.model.optimizer.lr, self.start_lr)
        n_steps = self.params['steps'] if self.params['steps'] is not None else round(
            self.params['samples'] / self.params['batch_size']
        )
        n_steps *= self.params['epochs']
        self.by = (self.end_lr - self.start_lr) / n_steps

    def on_batch_end(self, batch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
        self.lrs.append(lr)
        self.losses.append(logs.get('loss'))
        lr += self.by
        K.set_value(self.model.optimizer.lr, lr)


class SGDRSchedule(Callback):
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay,
                 cycle_length,
                 mult_factor):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.lr_decay = lr_decay
        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        self.history = {}

    def calc_lr(self):
        fraction_to_restart = self.batch_since_restart / (
            self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1 + np.cos(fraction_to_restart * np.pi)
        )
        return lr

    def on_train_begin(self, logs=None):
        logs = {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs=None):
        logs = {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.calc_lr())

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)


def step_decay_lr(initial_lr=1e-3, factor=0.2, step_size=2):

    def schedule(epoch):
        return initial_lr * (np.floor(epoch / step_size) ** factor)

    return LearningRateScheduler(schedule)


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

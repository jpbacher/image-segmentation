import numpy as np


def get_smooth_loss(loss_arr, points=20):
    b_point = np.ones(points) / points
    smooth_loss = np.convolve(loss_arr, b_point, mode='same')
    return smooth_loss


def get_lr_range(losses, points, finder_lrs):
    smooth_losses = get_smooth_loss(losses, points)
    smooth_diff = get_smooth_loss(np.diff(smooth_losses), 20)

    smooth_decrease = np.argmax(smooth_diff <= 0)
    smooth_increase = np.argmax(smooth_diff >= 0)
    max_ = smooth_increase if smooth_increase > 0 else smooth_diff.shape[0]

    smooth_losses = smooth_losses[smooth_decrease: max_]
    min_smooth_loss = min(smooth_losses[:-1])
    max_smooth_loss = max(smooth_losses[:-1])
    loss_delta = max_smooth_loss - min_smooth_loss

    lr_max = np.argmax(smooth_losses <= min_smooth_loss + 0.05 * loss_delta)
    lr_min = np.argmax(smooth_losses <= min_smooth_loss + 0.5 * loss_delta)
    lr_max += smooth_decrease
    lr_min += smooth_decrease

    lrs = finder_lrs[lr_min: lr_max]
    lr_min, lr_max = min(lrs), max(lrs)
    print(f'learning rate range: [{lr_min}, {lr_max}]')

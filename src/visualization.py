import numpy as np
import matplotlib.pyplot as plt


def plot_sample(car, mask):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].imshow(car)
    ax[0].axis('off')
    ax[0].title.set_text('Image')
    ax[1].imshow(mask)
    ax[1].axis('off')
    ax[1].title.set_text('Mask')
    plt.tight_layout()
    plt.show()


def smooth(y_arr, points=20):
    b_point = np.ones(points) / points
    smooth_y = np.convolve(y_arr, b_point, mode='same')
    return smooth_y


def plot_lr_finder(lrs, losses, points):
    smooth_losses = smooth(losses, points)
    plt.figure(figsize=(7, 5))
    plt.plot(lrs, smooth_losses)
    plt.xlabel('learning rate')
    plt.ylable('loss')
    plt.title('Smoothed Losses after Batch')
    plt.show()


def plot_losses(history):
    fix, ax = plt.subplots(2, 1, figsize=(8, 10))
    ax[0].plot(history['loss'], color='y', label='Training Loss')
    ax[0].plot(history['val_loss'], color='b', label='Val Loss')
    legend = ax[0].legend(loc='best', shadow=True)
    ax[1].plot(history['dice_coeff'], color='y', label='Training Dice')
    ax[1].plot(history['val_dice_coeff'], color='b', label='Val Dice')
    legend = ax[1].legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def make_inference(test_imgs, test_masks, predictions, num_samples):
    fig, ax = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    for i in range(num_samples):
        ax[i, 0].imshow(test_imgs[i].astype(np.float32))
        ax[i, 0].axis('off')
        ax[i, 0].title.set_text('Carvana Car')

        ax[i, 1].imshow(test_masks[i, :, :, 0].astype(np.float32), cmap='gray')
        ax[i, 1].axis('off')
        ax[i, 1].title.set_text('True Mask')

        ax[i, 2].imshow(predictions[i, :, :, 0], cmap='gray')
        ax[i, 2].axis('off')
        ax[i, 2].title.set_text('Predicted Mask')
    plt.show()

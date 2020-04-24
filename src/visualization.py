import matplotlib.pyplot as plt


def plot_img_mask(img, mask, input_size=480, cmap=None):
    fig, axs = plt.subplots(
        ncols=2, figsize=(18, 6), sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[1].imshow(mask[:, :], cmap=cmap)
    for ax in axs:
        ax.set_xlim(0, input_size)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

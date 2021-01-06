from matplotlib import pyplot as plt


def showInRow(imgs, names=None):
    fig, axs = plt.subplots(ncols=len(imgs), figsize=(16, 8))
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i])
        if names and i < len(names):
            ax.set_title(names[i], fontsize=15)
    plt.show()

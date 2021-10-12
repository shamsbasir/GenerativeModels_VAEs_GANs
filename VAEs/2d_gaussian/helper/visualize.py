import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#### Smiley dataset

def plot_scatter_2d(points, title='', labels=None):
    """
    Visualizes dataset
    """
    plt.figure()
    if labels is not None:
        plt.scatter(points[:, 0], points[:, 1], c=labels,
                    cmap = mpl.colors.ListedColormap(['red', 'blue', 'green', 'purple']))
    else:
        plt.scatter(points[:, 0], points[:, 1])
    plt.title(title)
    plt.show()


def visualize_samples(data, e):
    """
    e: int, epoch
    Visualizes how the network is learning in when training
    """
    plt.title('Sample Data Epoch: {}'.format(e))
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def plot_vae_training_plot(elbo, recon, kl, epochs, title):
    """
    Training and testing losses
    """
    plt.figure()
    x_train = np.linspace(0, epochs, len(elbo))

    plt.plot(x_train, elbo, label='-elbo_train')
    plt.plot(x_train, recon, label='recon_loss_train')
    plt.plot(x_train, kl, label='kl_loss_train')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
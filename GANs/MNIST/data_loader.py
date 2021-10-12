
# loading the data 
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import batch_size 
import torchvision
import torchvision.transforms as transforms



transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
)


train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True,drop_last=True,
)



if __name__ == "__main__":

	real_samples, mnist_labels = next(iter(train_loader))
	for i in range(16):
	    ax = plt.subplot(4, 4, i + 1)
	    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
	    plt.xticks([])
	    plt.yticks([])
	    plt.pause(1)


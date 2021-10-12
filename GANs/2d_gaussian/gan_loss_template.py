import torch
import torch.autograd as autograd 
import torch.nn as nn
from config import batch_size, Lambda


#Reference : https://deeplearning.cs.cmu.edu/S20/document/recitation/recitation13.pdf


class Generator(nn.Module):
    """
    Receives z and outputs x
    """
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Discriminator(nn.Module):
    """
    Receives x from real and from generated and outputs either 1 or 0
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x



def gan_loss_discriminator(Discriminator, Generator, x_real, z):

    """
    Implement original GAN's discriminator loss

    """
    batch_size = x_real.size(0)

    # 1) Cross entrop Loss between the predicted label D(x) and real label = 1

    D_real_loss =  torch.nn.BCELoss()(Discriminator(x_real) , torch.ones((batch_size, 1)))

    # 2) Cross entropy Loss between the predicted labels D(G(z)) and fake labels 0 

    D_fake_loss = torch.nn.BCELoss()(Discriminator(Generator(z)),torch.zeros((batch_size, 1)))

    return D_real_loss + D_fake_loss
    


def gan_loss_generator(Discriminator, Generator, z):
    """
    Implement original GAN's generator loss

    """
    batch_size = z.size(0)

    G_loss = -torch.nn.BCELoss()(Discriminator(Generator(z)),torch.zeros((batch_size, 1)))

    return G_loss



def non_saturating_gan_loss_generator(Discriminator, Generator, z):
    """
    Implement non-saturating version of GAN's generator loss
    """
    # When Discriminator is confident that Generator is fake, it will lead to G vanish

    # Cross Entropy loss between the predicted labels and real labels 

    G_loss = torch.nn.BCELoss()(Discriminator(Generator(z)),torch.ones((batch_size, 1)))

    return G_loss



def wgan_loss_discriminator(Discriminator, Generator, x_real, z):
    """
    Imeplement wgan discriminator loss
    """
    D_loss = - Discriminator(x_real).mean() + Discriminator(Generator(z)).mean()

    return D_loss

def wgan_gradient_penalty(Discriminator, real_data, fake_data):
    """
    Implement gradient penalty term
    """
    # Paper : Improved Training of Wasserstein GANS 
    # Algorith 1 :

    batch_size = real_data.size(0)

    epsilon    = torch.rand(batch_size,1)

    epsilon    = epsilon.expand(real_data.size())

    x_hat      = (epsilon * real_data + ((1 - epsilon)*fake_data)).requires_grad_(True)

    disc_x_hat = Discriminator(x_hat)

    gradients = autograd.grad(oudisc_x_hat,)

    gradients = autograd.grad(outputs=disc_x_hat, 
                              inputs=x_hat,
                              grad_outputs= torch.ones(disc_x_hat.size()),
                              create_graph=True, 
                              retain_graph=True,
                              only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty


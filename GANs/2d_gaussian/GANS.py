
# Reference :  https://deeplearning.cs.cmu.edu/S20/document/recitation/recitation13.pdf

import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from Discriminator import Discriminator
from Generator import Generator
from data_loader import train_loader
from gan_loss_template import gan_loss_discriminator
from gan_loss_template import non_saturating_gan_loss_generator

from config import lr, num_epochs, batch_size


# ----------------------------------------------
#  Models : discriminator and generatore 
# ----------------------------------------------

discriminator    = Discriminator()
generator        = Generator()


# ---------------
# optimizers
# ----------------

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator     = torch.optim.Adam(generator.parameters(), lr=lr)




# ------------
# training 
# --------------

for epoch in range(num_epochs):

    for n, (real_samples) in enumerate(train_loader):

        # Sample noise as generator input
        z = torch.randn((batch_size, 2))


        # -----------------
        #  Train Generator
        # -----------------

        # Training the generator
        optimizer_generator.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        loss_generator = non_saturating_gan_loss_generator(discriminator,generator,z)

        loss_generator.backward()

        optimizer_generator.step()



        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_discriminator.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        loss_discriminator = gan_loss_discriminator(discriminator,generator,real_samples,z)

        loss_discriminator.backward()

        optimizer_discriminator.step()


        # Show loss
        if (epoch % 50 == 0 ) and (n == batch_size - 1):
            print(f" Epoch : {epoch}, D Loss : {loss_discriminator} , G Loss : {loss_generator}")


# -----------------
#  checkpoints 
# ------------------

PATH = os.getcwd()+"/generator.pt"
torch.save(generator.state_dict(), PATH)

PATH = os.getcwd()+"/discriminator.pt"
torch.save(discriminator.state_dict(), PATH)


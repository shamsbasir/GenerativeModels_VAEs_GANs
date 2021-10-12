
import torch
from torch import nn
import matplotlib.pyplot as plt
from Generator import Generator
import os

# ----------------
# Loading the models 
# ------------------

generator     = Generator()
PATH = os.getcwd()+"/generator.pt"
generator.load_state_dict(torch.load(PATH))
generator.eval()

z = torch.randn(128, 2)
gen_sample = generator(z).detach()
plt.plot(gen_sample[:, 0], gen_sample[:, 1], ".")
plt.show()


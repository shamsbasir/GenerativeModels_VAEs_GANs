# Encoder 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self,zdim):
		super(Encoder,self).__init__()
		self.net = nn.Sequential(
      nn.Linear(784,256),
      nn.ELU(inplace=True),
      nn.Dropout(0.25),
      nn.Linear(256,256),
      nn.ELU(inplace=True),
      nn.Dropout(0.25),
      nn.Linear(256,128),
      nn.ELU(inplace=True)
    )
		self.mu 	  = nn.Linear(128,zdim)
		self.logvar = nn.Linear(128,zdim)


	def forward(self,x):
		x = self.net(x.view(-1,784))
		return self.mu(x), self.logvar(x)




if __name__ == '__main__':

	zdim   = 100
	encoder = Encoder(zdim)

	x = torch.randn(10,786)
	print(x.shape)
	mu,log_var = encoder(x)

	print(mu.shape)
	print(log_var.shape)




import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self,zdim):
        super(Decoder,self).__init__()
        self.net  = nn.Sequential(
            nn.Linear(zdim,256),
            nn.ELU(inplace=True),
            nn.Linear(256,256),
            nn.ELU(inplace=True),
            nn.Linear(256,784),
            nn.Tanh(), 
        )

    def forward(self,z):
        out = self.net(z)
        out = out.view(-1,1,28,28)
        return out
        

if __name__ == '__main__':
	
	zdim = 100
	z = torch.randn(10,zdim)
	print(z.shape)
	decoder = Decoder(zdim)

	out = decoder(z)
	print(out.shape)




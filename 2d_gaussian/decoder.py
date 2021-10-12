import torch
import torch.nn as nn
import torch.nn.functional as F



class Decoder(nn.Module):
    def __init__(self,zdim):
        super(Decoder,self).__init__()
        self.net  = nn.Sequential(
            nn.Linear(zdim,512),
            nn.ELU(),
            nn.Linear(512,256),
            nn.ELU(),
            nn.Linear(256,128),
            nn.ELU(),
            nn.Linear(128,2),
        )

    def forward(self,z):
        out = self.net(z)
        return out
        
        

if __name__ == '__main__':
	
	zdim = 2
	z = torch.randn(10,2)
	print(z.shape)
	decoder = Decoder(zdim)

	out = decoder(z)
	print(out.shape)




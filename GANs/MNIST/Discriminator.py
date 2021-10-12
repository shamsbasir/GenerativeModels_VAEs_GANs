# Discriminator 

from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output


if __name__=="__main__":

    D = Discriminator()

    print(D)
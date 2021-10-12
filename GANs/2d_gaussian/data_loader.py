
# loading the data 
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import batch_size 


train_data 	 = np.load('./data/train_data_gauss.npy')
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size =batch_size, 
    shuffle    =True,
    drop_last  =True
)









if __name__ == "__main__":
	for n, (real_data) in enumerate(train_loader):
		plt.plot(real_data[:,0],real_data[:,1],".")
		plt.draw()
		plt.pause(0.0001)
		plt.clf()

	plt.plot(train_data[:, 0], train_data[:, 1], ".")
	plt.show()




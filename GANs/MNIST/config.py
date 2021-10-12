#  Hyper parameters 

import torch 


batch_size = 64
lr 		   = 0.00015
Lambda 	   = 10
c 		   = 0.01
latent_dim = 100
n_critic   = 5 
n_epochs   = 150

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


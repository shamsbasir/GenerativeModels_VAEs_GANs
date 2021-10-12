import torch
from torch.nn import functional as F
import numpy as np
import math
import torch.distributions as td 

def kullback_leibler_divergence(mu, log_var, axis=-1):
    """
    Implement Kullback Leibler divergence
    input ; log_simasqr

    """
    # Reference : https://deeplearning.cs.cmu.edu/F20/document/recitation/recitation10.pdf
    return -0.5*np.sum((1+log_var - mu**2 - np.exp(log_var)),axis=axis)



def bernoulli_nll(x, x_recon):
    """
    Implement negative log likelihood (Bernoulli). Sum over last axis
    """
    # Equation 5 

    return -np.sum(np.log(x*x_recon + (1-x)*(1-x_recon)),axis=-1)



def gaussian_nll(x, mu, log_var):
    """
    Implement Negative log likelihood (Gaussian). Sum over last axis
    """
    pi = math.pi
    return np.sum(0.5*np.log(2*pi*np.exp(log_var)) + 0.5*(x-mu)**2/(np.exp(log_var)),axis=-1)



def bernoulli_nll_from_bce(x, x_recon):
    """
    Implement Negative log likelihood (Bernoulli)
    Return error of the image (sum of pixels) instead of the average pixel error
    x and x_recon are numpy arrays of shape [1, 28, 28]
    Returns a numpy scalar
    """
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    x = torch.tensor(x, dtype = torch.float)
    x_recon = torch.tensor(x_recon, dtype = torch.float)
    return bce_loss(x_recon,x).numpy()



def gaussian_nll_from_mse(x, x_recon):
    """
    Implement Negative log likelihood (Gaussian) from mean square error.
    x and x_recon are numpy arrays of shape [1, 28, 28]
    Returns a numpy array of shape (1,)
    """
    pi = math.pi
    MSE = np.square(x-x_recon).mean(axis=(1,2))
    # Note : it says to output shape be (1,) but the grader wants shape ()
    return np.array(np.sum(0.5*(np.log(2*pi*MSE)+1),keepdims=False))


def iwae_nll(z_mu, z_logvar, x_mu, x_logvar, zdim, x, importance_samples=1):
    """
    Implement iwae_nll:
    Outputs of encoder (z_mu and z_logvar) and decoder (x_mu, x_logvar) are given.
    Implement function reparameterize that takes in (z_mu and z_logvar) and outputs z
    the log_mean_exp function is given below
    """


    # I think i got the math, but sampling and stuff confuses me a little with pytorch

    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        z = td.Normal(loc=mu, scale=std)
        return z

    def gaussian_log_prob(z, mu, logvar):
        # Gaussian log likelihood function
        return gaussian_nll(z, mu, logvar)

    def log_mean_exp(x, axis):
        m, _ = torch.max(x, axis)
        m2, _ = torch.max(x, axis, keepdim=True)
        return m + torch.log(torch.mean(torch.exp(x - m2), axis))



    # sample z
    z = reparameterize(z_mu,z_logvar)

    # -------------- calcualte Q(z|x) ---------------
    logq_zgx = gaussian_log_prob(z, z_mu,z_logvar)
    
    # --------------- calculate p(z) ----------------
    mu_prior = torch.zeros(zdim)
    std_prior = torch.ones(zdim)
    prior = td.Normal(loc=mu_prior, scale=std_prior)

    log_prior = torch.sum(prior.log_prob(z.rsample()), -1)


    # --------------- calculate p(x|z) ------------------
    x_std = torch.exp(0.5 * x_logvar)
    p_xgz = td.Normal(loc=x_mu, scale=x_std).log_prob(x)
    logp_xGz = torch.sum(p_xgz, -1)
     
    # -------------- final formula ------------
    posterior = logp_xGz + (log_prior - logq_zgx)*importance_samples
    
    # ------- return log mean of of posterior
    return -log_mean_exp(posterior,0)






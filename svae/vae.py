"""Sub-module to define architectures of Variational Auto-Encoders"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.gamma import Gamma
import numpy as np

# Custome imports
from.sampling import sample_vmf, sample_gaussian
from.utils import Ive

def torch_gamma_func(val):
    """
    The gamma function in the PyTorch API is not directly accessible.
    To do so we need to use lgamma which computes ln(|gamma(val)|).
    Thus to access the gamma value we need to compose with the exponential.
    """
    return torch.exp(torch.lgamma(torch.tensor(val)))

class SVAE(nn.Module):
    """implémentation du s-vae avec distribution vmf"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # encodeur
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_kappa = nn.Linear(hidden_dim // 2, 1)
        # >> RAPH: I stand corrected, a Gaussian in Rd needs mu and sigma 
        # to be in Rd but that's not the case for the vMF: a vMF 
        # on Sd-1 (Rd) require mu in Rd but kappa in R ! 
        # So you were right :D
        
        # décodeur
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        # mu normalisé sur la sphère
        mu = self.fc_mu(h)
        mu = mu / (torch.norm(mu, dim=-1, keepdim=True) + 1e-8)
        
        # kappa positif
        kappa = F.softplus(self.fc_kappa(h)) +1
        # >> RAPH: Why +0.1 ? the authors used +1 => corrected
        
        return mu, kappa
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def kl_vmf(self, kappa):
        # >> RAPH: One of the main remarks in the calculations 
        # of the KL is that it does not depend on mu => removed unused mu
        """calcul de la divergence kl pour chaque sample du batch"""
        m = self.latent_dim
        
        # utilisation de ive pour stabilité numérique
        ive = Ive.apply(m/2, kappa)
        ive_prev = Ive.apply(m/2 - 1, kappa)
        # >> RAPH: Why go back to the iv instead of directly using ive
        # authors only use ive since exponentials will cancel out due to division
        # I removed the exponentials

        # >> RAPH: ive is not differentiable natively by PyTorch so it 
        # means calling ive on a detached tensor (no grad history) so that does not work
        # to do so we need to specify the backward ourselves (see p.14 equation 16)
        # implemented as Ive in utils.py 

        bessel_ratio = kappa * (ive / (ive_prev + 1e-8))
        # print("ive, ive_prev, kappa", ive, ive_prev, kappa)
        # terme log c_m(kappa)
        # >> RAPH: a true Iv term is necessary (not Ive because there is no ratio here, see Eq.4 p.3)
        iv = Ive.apply(m/2 - 1, kappa)
        # print("Iv, kappa", iv, kappa)
        log_cm = (m/2 - 1) * torch.log(kappa + 1e-8) - (m/2) * torch.log(2 * torch.tensor(torch.pi)) - (torch.log(iv + 1e-8))
        # >> RAPH: there was a kappa missing, I added it

        # terme constant
        const = (m/2) * torch.log(torch.tensor(torch.pi)) + torch.log(torch.tensor(2)) - torch.log(torch_gamma_func(m/2))
        return bessel_ratio + log_cm + const # see Eq.14 and Eq. 15 p.13
    
    def forward(self, x):
        # B x input_dim
        mu, kappa = self.encode(x)
        # B x latent_dim, B x 1

        # échantillonnage
        batch_size = x.shape[0]
        z = sample_vmf(mu, kappa, batch_size)
        # B x latent_dim
        
        # reconstruction
        x_recon = self.decode(z)
        # B x input_dim
        
        return x_recon, mu, kappa

    def reconstruction_loss(self, x_recon, x):
        # >> RAPH: The original x (not latent) are assumed to be Gaussian 
        # so MSE is good here (its implicit 
        # in the paper because they never speak about the prior on 
        # the input only the prior on the latent space)
        # By looking at their code we can see that they 
        # never actually code grep + gcor but only use BCE (they had a binary task
        # since they use the Binarized MNIST = their model predict logits for each 
        # pixel in the binary image)
        return F.mse_loss(x_recon, x)
    
    def full_step(self, x, beta_kl):
        x_recon, mu, kappa = self.forward(x)
            
        # reconstruction loss
        recon_loss = self.reconstruction_loss(x_recon, x)
        
        # kl divergence
        # >> RAPH: We average the kl loss over the batch (the usual 
        # sum used in the per term kl of the gaussian comes from an analytical 
        # formula that requires to sum over dimensions, but the overal KL loss
        # should be averaged for final loss computation)
        # Note: Sum and Means are both okay since its fndamentally a sum in both cases
        # however, a mean allows a KL term that is not dependent on the batch size whereas 
        # a sum is sensitive to this. Using a mean allows comparison across experiments of losses
        # whereas the sum does not.
        kl_loss = self.kl_vmf(kappa).mean()

        loss = recon_loss + beta_kl * kl_loss
        return loss, dict(recon=recon_loss.item(),
                          kl=kl_loss.item())
    
    def sample(self, mu, kappa):
        svae_latent_samples = []
        for i in range(mu.size(0)):
            z = sample_vmf(mu[i:i+1,:], kappa[i:i+1,:], 1)
            svae_latent_samples.append(z)
        return torch.cat(svae_latent_samples, dim=0)
    
    def get_latent_samples(self, data_tensor):
        with torch.no_grad():
            print("[SVAE] Encoding dataset..")
            mu_all, kappa_all = self.encode(data_tensor)

            print("[SVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            svae_latent_samples = self.sample(mu_all, kappa_all)
            svae_latent_samples = svae_latent_samples.cpu().numpy()
            return svae_latent_samples, mu_all, kappa_all
    
    def get_latent_distributions(self, data_tensor):
        with torch.no_grad():
            print("[SVAE] Encoding dataset..")
            mu_all, kappa_all = self.encode(data_tensor)
            svae_latent_dists = torch.cat([mu_all, kappa_all], dim=1)
            svae_latent_dists = svae_latent_dists.cpu().numpy()
            return svae_latent_dists, mu_all, kappa_all
    
    def get_latent(self, data_tensor, mode="sample"):
        if mode == "sample":
            return self.get_latent_samples(data_tensor)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")
    

class GaussianVAE(nn.Module):
    """vae standard avec prior gaussien"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # encodeur
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # décodeur
        self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return sample_gaussian(mu, std)
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def full_step(self, x, beta_kl):
        x_recon, mu, logvar = self.forward(x)
            
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()) 
        # >> RAPH: for each input we compute the formula for log q/p when q and p are gaussians
        kl_loss = kl_loss.mean()
        # >> RAPH: we then take the expected value (estimated over the batch)
        
        loss = recon_loss + beta_kl * kl_loss
        return loss, dict(recon=recon_loss.item(),
                          kl=kl_loss.item())

    def sample(self, mu, std):
        return sample_gaussian(mu, std)
    
    def get_latent_samples(self, data_tensor):
        with torch.no_grad():
            print("[NVAE] Encoding dataset..")
            mu_all, std_all = self.encode(data_tensor)

            print("[NVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            nvae_latent_samples = self.sample(mu_all, std_all)
            nvae_latent_samples = nvae_latent_samples.cpu().numpy()
            return nvae_latent_samples, mu_all, std_all
    
    def get_latent_distributions(self, data_tensor):
        with torch.no_grad():
            print("[NVAE] Encoding dataset..")
            mu_all, std_all = self.encode(data_tensor)
            nvae_latent_dists = torch.cat([mu_all, std_all], dim=1)
            nvae_latent_dists = nvae_latent_dists.cpu().numpy()
            return nvae_latent_dists, mu_all, std_all
    
    def get_latent(self, data_tensor, mode="sample"):
        if mode == "sample":
            return self.get_latent_samples(data_tensor)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")
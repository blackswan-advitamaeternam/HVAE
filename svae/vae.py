"""Sub-module to define architectures of Variational Auto-Encoders"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Custome imports
from.sampling import batch_sample_vmf, sample_gaussian
from.utils import Ive

def torch_gamma_func(val):
    """
    The gamma function in the PyTorch API is not directly accessible.
    To do so we need to use lgamma which computes ln(|gamma(val)|).
    Thus to access the gamma value we need to compose with the exponential.
    """
    return torch.exp(torch.lgamma(torch.tensor(val)))

def xavier_uniform_initialization(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class SVAE(nn.Module):
    """implémentation du s-vae avec distribution vmf"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, one_layer=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # encodeur
        if one_layer:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU())
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU())
        self.mu_encoder = nn.Linear(hidden_dim, latent_dim)
        self.kappa_encoder = nn.Linear(hidden_dim, 1)
        # >> RAPH: I stand corrected, a Gaussian in Rd needs mu and sigma 
        # to be in Rd but that's not the case for the vMF: a vMF 
        # on Sd-1 (Rd) require mu in Rd but kappa in R ! 
        # So you were right :D
        
        # >> RAPH: When kappa grows too high, the probability of acceptance drops
        # in Ulrich. Another paper from Nicols de Cao shows the unstability
        # see https://arxiv.org/pdf/2006.04437 Fig. 2
        # The relation Kappa vs Dim is not log linear but almost
        # By looking at it, for dim < 10000 a heuristic could be to cap the maximum
        # Kappa to 100 * dim
        self.max_kappa = latent_dim * 100

        # décodeur
        if one_layer:
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, input_dim))
        else:
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, input_dim)
                                            )
        # glorot and bengio init
        xavier_uniform_initialization(self.encoder)
        xavier_uniform_initialization(self.mu_encoder)
        xavier_uniform_initialization(self.kappa_encoder)
        xavier_uniform_initialization(self.decoder)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        h = self.encoder(x)
        # mu normalisé sur la sphère
        mu = self.mu_encoder(h)
        mu = mu / (torch.norm(mu, dim=-1, keepdim=True) + 1e-8)
        
        # kappa positif
        kappa = F.softplus(self.kappa_encoder(h)) + 1
        # >> RAPH: Why +0.1 ? the authors used +1 => corrected
        
        # >> RAPH: When kappa grows too high, the probability of acceptance drops
        # clipping to prevent getting stuck at sampling
        # not normalizing because that would completely change the resulting distribution we sample from !!
        kappa = torch.clip(kappa, max=self.max_kappa)

        return mu, kappa
    
    def decode(self, z):
        return self.decoder(z)
    
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
        ive = Ive.apply(m/2 - 1, kappa)
        # log iv = log ive + kappa
        log_iv = torch.log(ive + 1e-8) + kappa
        
        log_cm = (m/2 - 1) * torch.log(kappa + 1e-8) - (m/2) * torch.log(2 * torch.tensor(torch.pi)) - log_iv
        # >> RAPH: there was a kappa missing, I added it

        # terme constant
        const = (m/2) * torch.log(torch.tensor(torch.pi)) + torch.log(torch.tensor(2)) - torch.log(torch_gamma_func(m/2))
        return bessel_ratio + log_cm + const # see Eq.14 and Eq. 15 p.13
    
    def forward(self, x, return_latent=False):
        # B x input_dim
        mu, kappa = self.encode(x)
        # B x latent_dim, B x 1

        # échantillonnage
        batch_size = x.shape[0]
        z = self.sample(mu, kappa)
        # B x latent_dim
        
        # reconstruction
        x_recon = self.decode(z)
        # B x input_dim
        if return_latent:
            return x_recon, mu, kappa, z
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

        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
    def full_step(self, x, beta_kl, return_latent=False):
        if return_latent:
            x_recon, mu, kappa, latent = self.forward(x, return_latent)
        else:
            x_recon, mu, kappa = self.forward(x, return_latent)
            
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

        # LOSS = - ELBO = - (Recon - beta * KL)
        # BUT the MSE is already the (- recon) term when using Gaussian as the input prior
        loss = recon_loss + beta_kl * kl_loss
        if return_latent:
            return loss, dict(recon=recon_loss.detach(),
                            kl=kl_loss.detach()), latent
        return loss, dict(recon=recon_loss.detach(),
                          kl=kl_loss.detach())
    
    def marginal_ll_batch(self, x, N: int = 500):
        """
        Importance Weighted Autoencoder (Burda et al., 2016) estimator
        for a batch of samples. 
        """
        self.eval()
        with torch.no_grad():
            batch_size, data_dim = x.shape

            # Encode once for the whole batch
            mu, kappa = self.encode(x)   # (batch, latent_dim)

            latent_dim = mu.size(1)

            # Expand to (batch, N, latent_dim)
            mu = mu.unsqueeze(1).expand(batch_size, N, latent_dim)
            kappa = kappa.unsqueeze(1).expand(batch_size, N, 1)

            # Sample z for each datapoint and each draw
            print(f"Sampling {x.size(0)*N} samples..")
            z = self.sample(mu.reshape(-1, latent_dim), # (batch*N, latent_dim)
                            kappa.reshape(-1, 1)) 
            z = z.view(batch_size, N, latent_dim)

            # Decode to get distribution parameters for p(x|z)
            recon_mu = self.decode(z.reshape(-1, latent_dim))   # (batch*N, data_dim)
            recon_mu = recon_mu.view(batch_size, N, data_dim)

            # Likelihood term: log p(x|z)
            # Assume Gaussian with unit variance
            recon_dist = torch.distributions.Normal(loc=recon_mu, scale=1.0)
            log_p_x_z = recon_dist.log_prob(x.unsqueeze(1))     # (batch, 1, data_dim)
            log_p_x_z = log_p_x_z.sum(dim=2)                    # sum over data_dim => (batch, 1)

            # Prior term: log p(z) for uniform on sphere S^{latent_dim-1}
            log_p_z = (torch.lgamma(torch.tensor(latent_dim/2.0)) 
                    - (latent_dim/2.0)*torch.log(torch.tensor(2*torch.pi)))
            log_p_z = log_p_z.expand(batch_size, N)             # broadcast

            # Posterior term: log q(z|x) for vMF
            v = latent_dim/2 - 1
            ive = Ive.apply(v, kappa.reshape(-1, latent_dim))   # scaled Bessel
            log_iv = torch.log(ive + 1e-12) + kappa.reshape(-1, latent_dim)
            log_cm = (v * torch.log(kappa.reshape(-1, latent_dim) + 1e-12)
                    - (latent_dim/2) * torch.log(torch.tensor(2*torch.pi))
                    - log_iv)
            log_cm = log_cm.view(batch_size, N, -1).sum(dim=2)  # reduce over latent_dim

            dot = torch.sum(mu * z, dim=2) * kappa.squeeze(-1)  # (batch, N)
            log_q_z_x = log_cm + dot

            # IWAE weights
            log_w = log_p_x_z + log_p_z - log_q_z_x  # (batch, N)

            # Marginal log likelihood per datapoint
            log_px = torch.logsumexp(log_w, dim=1) - torch.log(torch.tensor(N))  # (batch,)

        return log_px  # size (batch,)


    def total_marginal_ll(self, tensor, N: int = 500, reduced: str = 'mean'):
        """
        Compute the total marginal log likelihood over a given dataset
        """
        datasetsize = tensor.size(0)
        print(f"[SVAE] Computing total marginal LL for {datasetsize} samples with {N} draws each..")
        LL = self.marginal_ll_batch(tensor, N=N)
        if reduced == 'sum':
            return LL.sum()
        return LL.mean()

    def sample(self, mu, kappa):
        return batch_sample_vmf(mu, kappa, mu.size(0))
        # old, non vectorized
        # svae_latent_samples = []
        # for i in range(mu.size(0)):
        #     z = sample_vmf(mu[i:i+1,:], kappa[i:i+1,:], 1)
        #     svae_latent_samples.append(z)
        # return torch.cat(svae_latent_samples, dim=0)
    
    def get_latent_samples(self, data_tensor, verbose=True):
        self.eval()
        with torch.no_grad():
            if verbose:
                print("[SVAE] Encoding dataset..")
            mu_all, kappa_all = self.encode(data_tensor)

            if verbose:
                print("[SVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            svae_latent_samples = self.sample(mu_all, kappa_all)
            svae_latent_samples = svae_latent_samples.cpu().numpy()
            return svae_latent_samples, mu_all, kappa_all
    
    def get_latent_distributions(self, data_tensor, verbose=True):
        self.eval()
        with torch.no_grad():
            if verbose:
                print("[SVAE] Encoding dataset..")
            mu_all, kappa_all = self.encode(data_tensor)
            svae_latent_dists = torch.cat([mu_all, kappa_all], dim=1)
            svae_latent_dists = svae_latent_dists.detach().cpu().numpy()
            return svae_latent_dists, mu_all, kappa_all
    
    def get_latent(self, data_tensor, mode="sample", verbose=True):
        if mode == "sample":
            return self.get_latent_samples(data_tensor, verbose)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor, verbose)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")

class GaussianVAE(nn.Module):
    """vae standard avec prior gaussien"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, one_layer=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # encodeur
        if one_layer:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU())
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU())
        self.mu_encoder = nn.Linear(hidden_dim, latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dim, latent_dim)
        
        # décodeur
        if one_layer:
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, input_dim))
        else:
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, input_dim)
                                            )
            
        # glorot and bengio init
        xavier_uniform_initialization(self.encoder)
        xavier_uniform_initialization(self.mu_encoder)
        xavier_uniform_initialization(self.logvar_encoder)
        xavier_uniform_initialization(self.decoder)

    @property
    def device(self):
        return next(self.parameters()).device
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu_encoder(h), self.logvar_encoder(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return sample_gaussian(mu, std)
    
    def forward(self, x, return_latent=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        if return_latent:
            return x_recon, mu, logvar, z
        return x_recon, mu, logvar
    
    def reconstruction_loss(self, x_recon, x):
        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
    def full_step(self, x, beta_kl, return_latent=False):
        if return_latent:
            x_recon, mu, logvar, latent = self.forward(x, return_latent)
        else:
            x_recon, mu, logvar = self.forward(x, return_latent)
            
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_loss = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()) 
        # >> RAPH: for each input we compute the formula for log q/p when q and p are gaussians
        kl_loss = kl_loss.sum(dim=1).mean()
        # >> RAPH: sum over dimensions, we then take the expected value (estimated over the batch)
        
        # LOSS = - ELBO = - (recon - beta * KL)
        # BUT the MSE is already the (- recon) term when using Gaussian as the input prior
        loss = recon_loss + beta_kl * kl_loss
        if return_latent:
            return loss, dict(recon=recon_loss.detach(),
                            kl=kl_loss.detach()), latent
        return loss, dict(recon=recon_loss.detach(),
                          kl=kl_loss.detach())
    
    def marginal_ll_batch(self, x, N: int = 500):
        """
        Importance Weighted Autoencoder (Burda et al., 2016) estimator
        for a batch of samples.
        """
        self.eval()
        with torch.no_grad():
            batch_size, data_dim = x.shape

            # Encode once for whole batch
            mu, logvar = self.encode(x)   # (batch, latent_dim)
            std = torch.exp(0.5 * logvar)

            latent_dim = mu.size(1)

            # Expand to (batch, N, latent_dim)
            mu = mu.unsqueeze(1).expand(batch_size, N, latent_dim)
            std = std.unsqueeze(1).expand(batch_size, N, latent_dim)

            # Sample z for each datapoint and each draw
            print(f"Sampling {x.size(0)*N} samples..")
            z = self.sample(mu.reshape(-1, latent_dim),
                        std.reshape(-1, latent_dim))   # (batch*N, latent_dim)
            z = z.view(batch_size, N, latent_dim)

            # Decode to get distribution parameters for p(x|z)
            recon_mu = self.decode(z.reshape(-1, latent_dim))   # (batch*N, data_dim)
            recon_mu = recon_mu.view(batch_size, N, data_dim)

            # Likelihood term: log p(x|z)
            # Assume Gaussian with unit variance
            recon_dist = torch.distributions.Normal(loc=recon_mu, scale=1.0)
            log_p_x_z = recon_dist.log_prob(x.unsqueeze(1))     # (batch, N, data_dim)
            log_p_x_z = log_p_x_z.sum(dim=2)                    # sum over data_dim => (batch, N)

            # Prior term: log p(z) under standard normal
            prior_dist = torch.distributions.Normal(
                loc=torch.zeros(latent_dim, device=x.device),
                scale=torch.ones(latent_dim, device=x.device)
            )
            log_p_z = prior_dist.log_prob(z)                    # (batch, N, latent_dim)
            log_p_z = log_p_z.sum(dim=2)                        # (batch, N)

            # Posterior term: log q(z|x) under encoder Gaussian
            approx_post = torch.distributions.Normal(loc=mu, scale=std)
            log_q_z_x = approx_post.log_prob(z)                 # (batch, N, latent_dim)
            log_q_z_x = log_q_z_x.sum(dim=2)                    # (batch, N)

            # IWAE weights
            log_w = log_p_x_z + log_p_z - log_q_z_x             # (batch, N)

            # Marginal log likelihood per datapoint
            log_px = torch.logsumexp(log_w, dim=1) - torch.log(torch.tensor(N))  # (batch,)

        return log_px  # size (batch,)


    def total_marginal_ll(self, tensor, N: int = 500, reduced: str = 'mean'):
        """
        Compute the total marginal log likelihood over a given dataset
        """
        datasetsize = tensor.size(0)
        print(f"[NVAE] Computing total marginal LL for {datasetsize} samples with {N} draws each..")
        LL = self.marginal_ll_batch(tensor, N=N)
        if reduced == 'sum':
            return LL.sum()
        return LL.mean()
    
    def sample(self, mu, std):
        return sample_gaussian(mu, std)
    
    def get_latent_samples(self, data_tensor, verbose=True):
        with torch.no_grad():
            if verbose:
                print("[NVAE] Encoding dataset..")
            mu_all, logvar_all = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

            if verbose:
                print("[NVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            nvae_latent_samples = self.sample(mu_all, std_all)
            nvae_latent_samples = nvae_latent_samples.detach().cpu().numpy()
            return nvae_latent_samples, mu_all, std_all
    
    def get_latent_distributions(self, data_tensor, verbose=True):
        with torch.no_grad():
            if verbose:
                print("[NVAE] Encoding dataset..")
            mu_all, logvar_all = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

            nvae_latent_dists = torch.cat([mu_all, std_all], dim=1)
            nvae_latent_dists = nvae_latent_dists.detach().cpu().numpy()
            return nvae_latent_dists, mu_all, std_all
    
    def get_latent(self, data_tensor, mode="sample", verbose=True):
        if mode == "sample":
            return self.get_latent_samples(data_tensor, verbose)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor, verbose)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")
        
class SVAE_M2(nn.Module):
    """implémentation du s-vae de type M2 avec distribution vmf"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_clusters, one_layer=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        # clustering block (outputs logits)
        self.cat_dist = torch.distributions.Categorical(torch.tensor([1/n_clusters]*n_clusters))

        if one_layer:
            self.cluster_block = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, n_clusters))
        else:
            self.cluster_block = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, n_clusters))
        # encodeur
        if one_layer:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU())
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU())
        self.mu_encoder = nn.Linear(hidden_dim+n_clusters, latent_dim)
        self.kappa_encoder = nn.Linear(hidden_dim, 1)
        # >> RAPH: I stand corrected, a Gaussian in Rd needs mu and sigma 
        # to be in Rd but that's not the case for the vMF: a vMF 
        # on Sd-1 (Rd) require mu in Rd but kappa in R ! 
        # So you were right :D
        
        # >> RAPH: When kappa grows too high, the probability of acceptance drops
        # in Ulrich. Another paper from Nicols de Cao shows the unstability
        # see https://arxiv.org/pdf/2006.04437 Fig. 2
        # The relation Kappa vs Dim is not log linear but almost
        # By looking at it, for dim < 10000 a heuristic could be to cap the maximum
        # Kappa to 100 * dim
        self.max_kappa = latent_dim * 100

        # décodeur
        if one_layer:
            self.decoder = nn.Sequential(nn.Linear(latent_dim+n_clusters, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, input_dim))
        else:
            self.decoder = nn.Sequential(nn.Linear(latent_dim+n_clusters, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, input_dim)
                                            )
        # glorot and bengio init
        xavier_uniform_initialization(self.cluster_block)
        xavier_uniform_initialization(self.encoder)
        xavier_uniform_initialization(self.mu_encoder)
        xavier_uniform_initialization(self.kappa_encoder)
        xavier_uniform_initialization(self.decoder)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        logits = self.cluster_block(x)
        
        h = self.encoder(x)
        
        # mu normalisé sur la sphère
        mu = self.mu_encoder(torch.cat([h,logits], dim=1))
        mu = mu / (torch.norm(mu, dim=-1, keepdim=True) + 1e-8)
        
        # kappa positif
        kappa = F.softplus(self.kappa_encoder(h)) + 1
        # >> RAPH: Why +0.1 ? the authors used +1 => corrected
        
        # >> RAPH: When kappa grows too high, the probability of acceptance drops
        # clipping to prevent getting stuck at sampling
        # not normalizing because that would completely change the resulting distribution we sample from !!
        kappa = torch.clip(kappa, max=self.max_kappa)

        return mu, kappa, logits
    
    def decode(self, logits, z):
        return self.decoder(torch.cat([z,logits], dim=1))
    
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
        ive = Ive.apply(m/2 - 1, kappa)
        # log iv = log ive + kappa
        log_iv = torch.log(ive + 1e-8) + kappa
        
        log_cm = (m/2 - 1) * torch.log(kappa + 1e-8) - (m/2) * torch.log(2 * torch.tensor(torch.pi)) - log_iv
        # >> RAPH: there was a kappa missing, I added it

        # terme constant
        const = (m/2) * torch.log(torch.tensor(torch.pi)) + torch.log(torch.tensor(2)) - torch.log(torch_gamma_func(m/2))
        return bessel_ratio + log_cm + const # see Eq.14 and Eq. 15 p.13
    
    def forward(self, x):
        # B x input_dim
        mu, kappa, logits = self.encode(x)
        # B x latent_dim, B x 1

        # échantillonnage
        batch_size = x.shape[0]
        z = self.sample(mu, kappa)
        # B x latent_dim
        
        # reconstruction
        x_recon = self.decode(logits, z)
        # B x input_dim
        
        return x_recon, mu, kappa, logits

    def reconstruction_loss(self, x_recon, x):
        # >> RAPH: The original x (not latent) are assumed to be Gaussian 
        # so MSE is good here (its implicit 
        # in the paper because they never speak about the prior on 
        # the input only the prior on the latent space)
        # By looking at their code we can see that they 
        # never actually code grep + gcor but only use BCE (they had a binary task
        # since they use the Binarized MNIST = their model predict logits for each 
        # pixel in the binary image)

        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
    def full_step(self, x, beta_kl, alpha):
        N = x.size(1)
        x_recon, mu, kappa, logits = self.forward(x)
            
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

        y_loss = self.cat_dist.log_prob(torch.argmax(logits, dim=1)) # per sample log prob
        y_loss = y_loss.mean() # expected value

        # LOSS = - ELBO = - (recon - beta * KL - alpha * N * KL_y) voir Kingma 2014 SemiSupervised
        loss = recon_loss + beta_kl * kl_loss + alpha * N * y_loss
        return loss, dict(recon=recon_loss.detach(),
                          kl=kl_loss.detach(),
                          y_loss=y_loss.detach())

    def sample(self, mu, kappa):
        return batch_sample_vmf(mu, kappa, mu.size(0))
    
    def get_latent_samples(self, data_tensor, verbose=True):
        self.eval()
        with torch.no_grad():
            if verbose:
                print("[SVAE] Encoding dataset..")
            mu_all, kappa_all, logits = self.encode(data_tensor)

            if verbose:
                print("[SVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            svae_latent_samples = self.sample(mu_all, kappa_all)
            svae_latent_samples = svae_latent_samples.detach().cpu().numpy()
            return svae_latent_samples, mu_all, kappa_all, logits
    
    def get_latent_distributions(self, data_tensor, verbose=True):
        self.eval()
        with torch.no_grad():
            if verbose:
                print("[SVAE] Encoding dataset..")
            mu_all, kappa_all, logits = self.encode(data_tensor)
            svae_latent_dists = torch.cat([mu_all, kappa_all], dim=1)
            svae_latent_dists = svae_latent_dists.detach().cpu().numpy()
            return svae_latent_dists, mu_all, kappa_all, logits
    
    def get_latent(self, data_tensor, mode="sample", verbose=True):
        if mode == "sample":
            return self.get_latent_samples(data_tensor, verbose)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor, verbose)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")
        
class GaussianVAE_M2(nn.Module):
    """
    vae de type M2 avec prior gaussien (z et y sont des variables latentes)
    mu depends de y et de z
    sigma depends de z uniquement
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, n_clusters, one_layer=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        # clustering block (outputs logits)
        self.cat_dist = torch.distributions.Categorical(torch.tensor([1/n_clusters]*n_clusters))

        if one_layer:
            self.cluster_block = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, n_clusters))
        else:
            self.cluster_block = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, n_clusters))
            
        # encodeur
        if one_layer:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        nn.ReLU())
        else:
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, hidden_dim),
                                            nn.ReLU())
        self.mu_encoder = nn.Linear(hidden_dim+n_clusters, latent_dim) # depends on y also
        self.logvar_encoder = nn.Linear(hidden_dim, latent_dim)
        
        # décodeur (depends de y et z)
        if one_layer:
            self.decoder = nn.Sequential(nn.Linear(latent_dim+n_clusters, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, input_dim))
        else:
            self.decoder = nn.Sequential(nn.Linear(latent_dim+n_clusters, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim * 2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim * 2, input_dim)
                                            )
            
        # glorot and bengio init
        xavier_uniform_initialization(self.cluster_block)
        xavier_uniform_initialization(self.encoder)
        xavier_uniform_initialization(self.mu_encoder)
        xavier_uniform_initialization(self.logvar_encoder)
        xavier_uniform_initialization(self.decoder)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def encode(self, x):
        h = self.encoder(x)
        logits = self.cluster_block(x)
        return self.mu_encoder(torch.cat([h,logits], dim=1)), self.logvar_encoder(h), logits
    
    def decode(self, logits, z):
        return self.decoder(torch.cat([z,logits], dim=1))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return sample_gaussian(mu, std)
    
    def forward(self, x):
        mu, logvar, logits = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(logits, z)
        return x_recon, mu, logvar, logits
    
    def reconstruction_loss(self, x_recon, x):
        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
    def full_step(self, x, beta_kl, alpha):
        N = x.size(1)
        x_recon, mu, logvar, logits = self.forward(x)
            
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_loss = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()) 
        # >> RAPH: for each input we compute the formula for log q/p when q and p are gaussians
        kl_loss = kl_loss.sum(dim=1).mean()
        # >> RAPH: sum over dimensions, we then take the expected value (estimated over the batch)
        
        y_loss = self.cat_dist.log_prob(torch.argmax(logits, dim=1)) # per sample log prob
        y_loss = y_loss.mean() # expected value

        # LOSS = - ELBO = - (recon - beta * KL + alpha * N * KL_y) voir Kingma 2014 SemiSupervised
        loss = recon_loss + beta_kl * kl_loss - alpha * N * y_loss
        return loss, dict(recon=recon_loss.detach(),
                          kl=kl_loss.detach(),
                          y_loss=y_loss.detach())
    
    def sample(self, mu, std):
        return sample_gaussian(mu, std)
    
    def get_latent_samples(self, data_tensor, verbose=True):
        with torch.no_grad():
            if verbose:
                print("[NVAE] Encoding dataset..")
            mu_all, logvar_all, logits = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

            if verbose:
                print("[NVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            nvae_latent_samples = self.sample(mu_all, std_all)
            nvae_latent_samples = nvae_latent_samples.detach().cpu().numpy()
            return nvae_latent_samples, mu_all, std_all, logits
    
    def get_latent_distributions(self, data_tensor, verbose=True):
        with torch.no_grad():
            if verbose:
                print("[NVAE] Encoding dataset..")
            mu_all, logvar_all, logits = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

            nvae_latent_dists = torch.cat([mu_all, std_all], dim=1)
            nvae_latent_dists = nvae_latent_dists.detach().cpu().numpy()
            return nvae_latent_dists, mu_all, std_all, logits
    
    def get_latent(self, data_tensor, mode="sample", verbose=True):
        if mode == "sample":
            return self.get_latent_samples(data_tensor, verbose)
        elif mode == "dist":
            return self.get_latent_distributions(data_tensor, verbose)
        else:
            raise ValueError(f"Unrecognized mode {mode}. Should either be 'sample' or 'dist'.")

def arccos(x, y):
    sim = np.clip(x @ y, -1.0, 1.0)
    return np.arccos(sim)

def arccos_with_grad(x, y, eps=1e-8):
    """
    Angular distance based on cosine similarity for unnormalized vectors.
    Returns:
        dist: float
        grad: np.ndarray  (gradient w.r.t. x)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    a = np.dot(x, y)                              # x·y
    bx = np.linalg.norm(x) + eps                  # ||x||
    by = np.linalg.norm(y) + eps                  # ||y||

    s = a / (bx * by)                             # cosine similarity
    s = np.clip(s, -1.0 + eps, 1.0 - eps)         # keep arccos stable

    dist = np.arccos(s)

    # ds/dx = (1/(by*bx)) * ( y - (a/bx^2) * x )
    ds_dx = (y - (a / (bx * bx)) * x) / (by * bx)

    # d/dx arccos(s) = -(1/sqrt(1-s^2)) * ds/dx
    grad = -ds_dx / (np.sqrt(1.0 - s * s) + eps)

    return dist, grad
    
class M1:
    def __init__(
        self,
        vae_type,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        one_layer: bool = True,
        **kwargs # for the KNN
        ):
        self.latent_dim = latent_dim

        # Inner VAE (Gaussian or SVAE)
        if vae_type == "normal":
            vae_cls = GaussianVAE
        elif vae_type == "svae":
            vae_cls = SVAE
            if "metric" not in kwargs.keys():
                kwargs["metric"] = arccos
        else:
            raise ValueError(f"Unknown type '{vae_type}', must be either 'normal' or 'svae'.")
        
        self.vae = vae_cls(input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           latent_dim=latent_dim,
                           one_layer=one_layer)
        
        self.clf = KNeighborsClassifier(**kwargs)
        self._clf_is_fitted = False
    
    def to(self, device):
        self.vae.to(device)
        
    def __getattr__(self, name):
        try:
            return getattr(self.vae, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object and its 'vae' attribute have no attribute '{name}'")
        
    def fit_clf(self, data_tensor, label_tensor, mode):
        latent, _, _ = self.get_latent(data_tensor, mode)
        self.clf.fit(latent, label_tensor.detach().cpu().numpy())
        self._clf_is_fitted = True
        return self
    
    def predict_class(self, data_tensor, mode, return_latent=False):
        assert self._clf_is_fitted, f"Classifier not fitted yet. you must call fit_clf beforehand."
        latent, _, _ = self.get_latent(data_tensor, mode, verbose=False)
        if return_latent:
            return self.clf.predict(latent), latent
        return self.clf.predict(latent)

class M1_M2:
    def __init__(self,
        m1_type,
        m2_type,
        input_dim: int,
        hidden_dim: int,
        latent_dim1: int,
        latent_dim2: int,
        n_clusters: int,
        one_layer: bool = True,
        ):
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2

        # M1
        if m1_type == "normal":
            vae_m1 = GaussianVAE
        elif m1_type == "svae":
            vae_m1 = SVAE
        else:
            raise ValueError(f"Unknown type '{m1_type}', must be either 'normal' or 'svae'.")
        
        self.vae_m1 = vae_m1(input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           latent_dim=latent_dim1,
                           one_layer=one_layer)
        
        # M2
        if m2_type == "normal":
            vae_m2 = GaussianVAE_M2
        elif m2_type == "svae":
            vae_m2 = SVAE_M2
        else:
            raise ValueError(f"Unknown type '{m2_type}', must be either 'normal' or 'svae'.")
        
        self.vae_m2 = vae_m2(input_dim=latent_dim1,
                           hidden_dim=hidden_dim,
                           latent_dim=latent_dim2,
                           n_clusters=n_clusters,
                           one_layer=one_layer)

    def to(self, device):
        self.vae_m1.to(device)
        self.vae_m2.to(device)

    def __getattr__(self, name):
        try:
            if 'm1' in name:
                return getattr(self.vae_m1, name)
            elif 'm2' in name:
                return getattr(self.vae_m2, name)
            elif name == 'parameters':
                param_gen_m1 = self.vae_m1.parameters()
                param_gen_m2 = self.vae_m2.parameters()
                def m1m2_params():
                    yield from param_gen_m1
                    yield from param_gen_m2
                return m1m2_params
            elif name == 'train':
                def set_train(mode: bool = True):
                    self.vae_m1.train(mode)
                    self.vae_m2.train(mode)
                    return self
                return set_train
            elif name == 'eval':
                def set_eval():
                    self.vae_m1.eval()
                    self.vae_m2.eval()
                    return self
                return set_eval

            elif name == "state_dict":
                # Return a callable like torch.nn.Module.state_dict
                def _state_dict(*args, **kwargs):
                    return {
                        "vae_m1": self.vae_m1.state_dict(*args, **kwargs),
                        "vae_m2": self.vae_m2.state_dict(*args, **kwargs),
                        # optional: store meta to help debugging / compatibility
                        "meta": {
                            "latent_dim1": self.latent_dim1,
                            "latent_dim2": self.latent_dim2,
                        },
                    }
                return _state_dict

            elif name == "load_state_dict":
                def _load_state_dict(state, strict: bool = True):
                    """
                    Supports both:
                    - nested dict: {"vae_m1": ..., "vae_m2": ...}
                    - flat/prefixed dict: {"vae_m1.<k>": ..., "vae_m2.<k>": ...}
                    Returns something similar to nn.Module.load_state_dict:
                    (missing_keys, unexpected_keys)
                    """
                    # Case 1: nested format
                    if isinstance(state, dict) and "vae_m1" in state and "vae_m2" in state:
                        missing_all = []
                        unexpected_all = []

                        out1 = self.vae_m1.load_state_dict(state["vae_m1"], strict=strict)
                        out2 = self.vae_m2.load_state_dict(state["vae_m2"], strict=strict)

                        # PyTorch returns either None or an IncompatibleKeys object
                        if out1 is not None:
                            missing_all += list(getattr(out1, "missing_keys", []))
                            unexpected_all += list(getattr(out1, "unexpected_keys", []))
                        if out2 is not None:
                            missing_all += list(getattr(out2, "missing_keys", []))
                            unexpected_all += list(getattr(out2, "unexpected_keys", []))

                        return missing_all, unexpected_all

                    # Case 2: flat/prefixed format
                    m1_sd = {}
                    m2_sd = {}
                    meta = None
                    for k, v in state.items():
                        if k.startswith("vae_m1."):
                            m1_sd[k[len("vae_m1."):]] = v
                        elif k.startswith("vae_m2."):
                            m2_sd[k[len("vae_m2."):]] = v
                        elif k == "meta":
                            meta = v

                    missing_all = []
                    unexpected_all = []

                    out1 = self.vae_m1.load_state_dict(m1_sd, strict=strict)
                    out2 = self.vae_m2.load_state_dict(m2_sd, strict=strict)

                    if out1 is not None:
                        missing_all += [f"vae_m1.{k}" for k in getattr(out1, "missing_keys", [])]
                        unexpected_all += [f"vae_m1.{k}" for k in getattr(out1, "unexpected_keys", [])]
                    if out2 is not None:
                        missing_all += [f"vae_m2.{k}" for k in getattr(out2, "missing_keys", [])]
                        unexpected_all += [f"vae_m2.{k}" for k in getattr(out2, "unexpected_keys", [])]

                    return missing_all, unexpected_all

                return _load_state_dict
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object and its 'vae_m1' and 'vae_m2' attribute have no attribute '{name}'")
        
    def encode_M1(self, x):
        return self.vae_m1.encode(x)

    def decode_M1(self, z):
        return self.vae_m1.decode(z)

    def forward_M1(self, x):
        param1, param2 = self.encode_M1(x)
        z = self.sample(param1, param2)
        x_recon = self.decode(z)
        return x_recon, param1, param2
    
    def encode_M2(self, z1):
        return self.vae_m2.encode(z1)

    def decode_M2(self, logits, z2):
        return self.vae_m2.decode(logits, z2)

    def forward_M2(self, z1):
        param1, param2, logits = self.encode_M2(z1)
        z = self.sample(param1, param2)
        x_recon = self.decode(logits, z)
        return x_recon, param1, param2, logits
    
    def forward(self, x):
        param1_M1, param2_M1 = self.encode_M1(x)
        z1 = self.sample(param1_M1, param2_M1)
        x_recon = self.decode(z1)

        param1_M2, param2_M2, logits = self.encode_M2(z1)
        z2 = self.sample(param1_M2, param2_M2)
        z1_recon = self.decode(logits, z2)
        return x_recon, param1_M1, param2_M1, z1_recon, param1_M2, param2_M2, logits

    def full_step(self, x, beta_kl, alpha):
        M1_loss, M1_dict, z1 = self.vae_m1.full_step(x, beta_kl, return_latent=True)
        M2_loss, M2_dict = self.vae_m2.full_step(z1, beta_kl, alpha)

        loss = M1_loss + M2_loss
        return loss, dict(M1=M1_loss.detach(),
                          M2=M2_loss.detach())
    
    def predict_class(self, data_tensor, mode, return_latent=False):
        latent_M1, _, _ = self.vae_m1.get_latent(data_tensor, mode, verbose=False)
        latent_M2, _, _, logits = self.vae_m2.get_latent(torch.from_numpy(latent_M1), mode, verbose=False)
        if return_latent:
            return torch.argmax(logits, dim=1).detach().cpu().numpy(), latent_M1, latent_M2
        return torch.argmax(logits, dim=1).detach().cpu().numpy()
    
def predict_classes_loader(model, loader, mode, return_latent=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Y = []
    Y_hat = []
    if return_latent and isinstance(model, M1_M2):
        Z1 = []
        Z2 = []
    if return_latent and isinstance(model, M1):
        Z1 = []

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        if return_latent and isinstance(model, M1_M2):
            y_hat, z1, z2 = model.predict_class(x, mode, return_latent)
            Z1.append(z1)
            Z2.append(z2)
        elif return_latent and isinstance(model, M1):
            y_hat, z1 = model.predict_class(x, mode, return_latent)
            Z1.append(z1)
        else:
            y_hat = model.predict_class(x, mode, return_latent)

        Y.append(y.detach().cpu().numpy())
        Y_hat.append(y_hat)
    
    Y = np.concat(Y)
    Y_hat = np.concat(Y_hat)
    if return_latent and isinstance(model, M1_M2):
        Z1 = np.concat(Z1)
        Z2 = np.concat(Z2)
        return Y, Y_hat, Z1, Z2
    elif return_latent and isinstance(model, M1):
        Z1 = np.concat(Z1)
        return Y, Y_hat, Z1
    return Y, Y_hat
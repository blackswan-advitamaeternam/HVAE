"""Sub-module to define architectures of Variational Auto-Encoders"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# class SVAE(nn.Module):
#     """implémentation du s-vae avec distribution vmf"""
    
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super().__init__()
#         self.latent_dim = latent_dim
        
#         # encodeur
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
#         self.fc_kappa = nn.Linear(hidden_dim // 2, 1)
#         # >> RAPH: I stand corrected, a Gaussian in Rd needs mu and sigma 
#         # to be in Rd but that's not the case for the vMF: a vMF 
#         # on Sd-1 (Rd) require mu in Rd but kappa in R ! 
#         # So you were right :D
        
#         # >> RAPH: When kappa grows too high, the probability of acceptance drops
#         # in Ulrich. Another paper from Nicols de Cao shows the unstability
#         # see https://arxiv.org/pdf/2006.04437 Fig. 2
#         # The relation Kappa vs Dim is not log linear but almost
#         # By looking at it, for dim < 10000 a heuristic could be to cap the maximum
#         # Kappa to 100 * dim
#         self.max_kappa = latent_dim * 100

#         # décodeur
#         self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
#         self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
#         self.fc5 = nn.Linear(hidden_dim, input_dim)
        
#     def encode(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
        
#         # mu normalisé sur la sphère
#         mu = self.fc_mu(h)
#         mu = mu / (torch.norm(mu, dim=-1, keepdim=True) + 1e-8)
        
#         # kappa positif
#         kappa = F.softplus(self.fc_kappa(h)) +1
#         # >> RAPH: Why +0.1 ? the authors used +1 => corrected
        
#         # >> RAPH: When kappa grows too high, the probability of acceptance drops
#         # clipping to prevent getting stuck at sampling
#         # not normalizing because that would completely change the resulting distribution we sample from !!
#         kappa = torch.clip(kappa, max=self.max_kappa)

#         return mu, kappa
    
#     def decode(self, z):
#         h = F.relu(self.fc3(z))
#         h = F.relu(self.fc4(h))
#         return self.fc5(h)
class SVAE(nn.Module):
    """implémentation du s-vae avec distribution vmf"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # encodeur
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_kappa = nn.Linear(hidden_dim, 1)
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
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, input_dim))
        
    def encode(self, x):
        h = F.relu(self.encoder(x))
        
        # mu normalisé sur la sphère
        mu = self.fc_mu(h)
        mu = mu / (torch.norm(mu, dim=-1, keepdim=True) + 1e-8)
        
        # kappa positif
        kappa = F.softplus(self.fc_kappa(h)) +1
        # >> RAPH: Why +0.1 ? the authors used +1 => corrected
        
        # >> RAPH: When kappa grows too high, the probability of acceptance drops
        # clipping to prevent getting stuck at sampling
        # not normalizing because that would completely change the resulting distribution we sample from !!
        kappa = torch.clip(kappa, max=self.max_kappa)

        return mu, kappa
    
    def decode(self, z):
        return F.relu(self.decoder(z))
    
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
        mu, kappa = self.encode(x)
        # B x latent_dim, B x 1

        # échantillonnage
        batch_size = x.shape[0]
        z = self.sample(mu, kappa)
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

        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
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

        # LOSS = - ELBO = - (Recon - beta * KL)
        # BUT the MSE is already the (- recon) term when using Gaussian as the input prior
        loss = recon_loss + beta_kl * kl_loss
        return loss, dict(recon=recon_loss.item(),
                          kl=kl_loss.item())
    
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
    
    def get_latent_samples(self, data_tensor):
        self.eval()
        with torch.no_grad():
            print("[SVAE] Encoding dataset..")
            mu_all, kappa_all = self.encode(data_tensor)

            print("[SVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            svae_latent_samples = self.sample(mu_all, kappa_all)
            svae_latent_samples = svae_latent_samples.cpu().numpy()
            return svae_latent_samples, mu_all, kappa_all
    
    def get_latent_distributions(self, data_tensor):
        self.eval()
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
    

# class GaussianVAE(nn.Module):
#     """vae standard avec prior gaussien"""
    
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super().__init__()
        
#         # encodeur
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
#         # décodeur
#         self.fc3 = nn.Linear(latent_dim, hidden_dim // 2)
#         self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim)
#         self.fc5 = nn.Linear(hidden_dim, input_dim)
        
#     def encode(self, x):
#         h = F.relu(self.fc1(x))
#         h = F.relu(self.fc2(h))
#         return self.fc_mu(h), self.fc_logvar(h)
    
#     def decode(self, z):
#         h = F.relu(self.fc3(z))
#         h = F.relu(self.fc4(h))
#         return self.fc5(h)

class GaussianVAE(nn.Module):
    """vae standard avec prior gaussien"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # encodeur
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # décodeur
        self.decoder = self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, input_dim))
        
    def encode(self, x):
        h = F.relu(self.encoder(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode(self, z):
        return F.relu(self.decoder(z))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return sample_gaussian(mu, std)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def reconstruction_loss(self, x_recon, x):
        # dim squared error
        se = (x_recon - x) ** 2  
        # sum over dimensions, mean over batch
        return se.view(se.size(0), -1).sum(dim=1).mean()
    
    def full_step(self, x, beta_kl):
        x_recon, mu, logvar = self.forward(x)
            
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_loss = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()) 
        # >> RAPH: for each input we compute the formula for log q/p when q and p are gaussians
        kl_loss = kl_loss.sum(dim=1).mean()
        # >> RAPH: sum over dimensions, we then take the expected value (estimated over the batch)
        
        # LOSS = - ELBO = - (recon - beta * KL)
        # BUT the MSE is already the (- recon) term when using Gaussian as the input prior
        loss = recon_loss + beta_kl * kl_loss
        return loss, dict(recon=recon_loss.item(),
                          kl=kl_loss.item())
    
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
    
    def get_latent_samples(self, data_tensor):
        with torch.no_grad():
            print("[NVAE] Encoding dataset..")
            mu_all, logvar_all = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

            print("[NVAE] Sampling from latent space..")
            # For each input's latent distribution, sample 1 element
            nvae_latent_samples = self.sample(mu_all, std_all)
            nvae_latent_samples = nvae_latent_samples.cpu().numpy()
            return nvae_latent_samples, mu_all, std_all
    
    def get_latent_distributions(self, data_tensor):
        with torch.no_grad():
            print("[NVAE] Encoding dataset..")
            mu_all, logvar_all = self.encode(data_tensor)
            std_all = torch.exp(0.5 * logvar_all)

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


class UnsupervisedClusteringVAE(nn.Module):
    """
    Wrapper around GaussianVAE or SVAE that:
      - trains the inner VAE unsupervised (standard ELBO)
      - adds a clustering head q(y | z) over K clusters
      - provides a utility method to compute the unsupervised loss

    The clustering head is intended for:
      - inducing cluster structure in the latent space
      - later using q(y | z) as an unsupervised classifier

    This is an M1-style model with an additional clustering regularizer,
    not the full M2 semi-supervised generative model.
    """

    def __init__(
        self,
        vae_cls,               # GaussianVAE or SVAE class
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_clusters: int,
        beta_kl: float = 1.0,
    ):
        """
        Args
        ----
        vae_cls: class
            Either GaussianVAE or SVAE (or any compatible VAE with full_step).
        input_dim: int
            Dimension of the observed data x.
        hidden_dim: int
            Hidden dimension for the inner VAE.
        latent_dim: int
            Latent dimension of the inner VAE.
        num_clusters: int
            Assumed number of clusters K in the latent space.
        beta_kl: float
            Weight on the KL term in the inner VAE loss (beta-VAE style).
        """
        super().__init__()

        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl

        # Inner VAE (Gaussian or SVAE)
        self.vae = vae_cls(input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           latent_dim=latent_dim)

        # Simple clustering head: q(y | z)
        # We use the latent mean as features (mu for Gaussian, mu for SVAE).
        self.cluster_head = nn.Linear(latent_dim, num_clusters)

    def encode(self, x):
        """
        Convenience pass-through to the inner VAE encoder.
        Returns whatever the inner encoder returns.
        """
        return self.vae.encode(x)

    def forward(self, x):
        """
        Forward pass through the inner VAE only.
        (We keep this as a thin wrapper around the base model.)
        """
        return self.vae(x)

    def _cluster_logits(self, latent_mean):
        """
        Compute logits for q(y | z) from the latent mean.
        """
        return self.cluster_head(latent_mean)

    def unsupervised_loss(self, x, entropy_weight: float = 1.0):
        """
        Compute unsupervised loss:

            L_total = L_vae(x) + entropy_weight * L_cluster(x)

        where
            L_vae(x)       = recon + beta_kl * KL   (from inner VAE.full_step)
            L_cluster(x)   = KL( \bar{q}(y) || Uniform(K) ) - H( q(y|z) )

        - The first term encourages good reconstruction + regularized latent.
        - The clustering regularizer encourages:
            * Balanced use of all K clusters (KL to uniform)
            * Non-degenerate assignments per sample (via negative entropy)

        Returns
        -------
        total_loss: torch.Tensor (scalar)
        logs: dict of scalars for monitoring
        """
        # ----- 1. Inner VAE loss (unsupervised ELBO) -----
        vae_loss, vae_logs = self.vae.full_step(x, beta_kl=self.beta_kl)
        # vae_loss: scalar, already recon + beta_kl * kl

        # We also need the latent mean to build q(y | z).
        # GaussianVAE.encode -> (mu, logvar)
        # SVAE.encode        -> (mu, kappa)
        with torch.no_grad():
            enc_out = self.vae.encode(x)
        latent_mean = enc_out[0]   # first element is mu in both models

        # ----- 2. Clustering head and regularizer -----
        logits_y = self._cluster_logits(latent_mean)          # (B, K)
        q_y = F.softmax(logits_y, dim=-1)                     # (B, K)

        # (a) Encourage balanced clusters across the batch:
        #     KL( \bar{q}(y) || Uniform(K) )
        batch_mean = q_y.mean(dim=0)                          # (K,)
        prior = torch.full_like(batch_mean, 1.0 / self.num_clusters)
        kl_to_uniform = torch.sum(
            batch_mean * (torch.log(batch_mean + 1e-8) - torch.log(prior + 1e-8))
        )

        # (b) Encourage confident assignments per sample:
        #     - H(q(y|z)) = sum q(y|z) log q(y|z) (averaged over batch)
        entropy = - (q_y * torch.log(q_y + 1e-8)).sum(dim=1).mean()

        # Combined clustering regularizer
        cluster_reg = kl_to_uniform - entropy

        # ----- 3. Total loss -----
        total_loss = vae_loss + entropy_weight * cluster_reg

        logs = {
            "loss_total": total_loss.item(),
            "loss_vae": vae_loss.item(),
            "cluster_reg": cluster_reg.item(),
            "recon": vae_logs.get("recon", None),
            "kl_latent": vae_logs.get("kl", None),
            "kl_to_uniform": kl_to_uniform.item(),
            "entropy_q_y": entropy.item(),
        }

        return total_loss, logs

    def predict_clusters(self, x, use_mean: bool = True):
        """
        Predict cluster assignments for x using q(y | z).

        Returns:
            hard_assignments: (B,) long tensor of argmax cluster indices
            probs:            (B, K) soft cluster probabilities
        """
        with torch.no_grad():
            enc_out = self.vae.encode(x)
            latent_mean = enc_out[0]
            logits_y = self._cluster_logits(latent_mean)
            probs = F.softmax(logits_y, dim=-1)
            hard = probs.argmax(dim=-1)
        return hard, probs
"""Sub-module to define sampling procedures and utility functions"""
import torch

# fonctions utilitaires pour vmf
def sample_vmf(mu, kappa, batch_size):
    """échantillonnage depuis vmf en dimension m"""
    m = mu.shape[-1]
    
    # cas 2d simplifié
    if m == 2:
        angles = torch.randn(batch_size, 1) * 2 * torch.pi
        x = torch.cos(angles)
        y = torch.sin(angles)
        samples = torch.cat([x, y], dim=1)
        
        # concentration autour de mu
        for i in range(batch_size):
            if kappa[i] > 0:
                mu_angle = torch.atan2(mu[i,1], mu[i,0])
                concentrated_angles = mu_angle + torch.randn(1, 1) / (kappa[i] + 1e-8)
                xi = torch.cos(concentrated_angles)
                yi = torch.sin(concentrated_angles)
                samples[i] = torch.cat([xi, yi], dim=1)
        return samples
    
    # cas général - algorithme d'ulrich
    b = -2 * kappa + torch.sqrt(4 * kappa**2 + (m-1)**2)
    b = b / (m - 1)
    a = (m - 1 + 2 * kappa + torch.sqrt(4 * kappa**2 + (m-1)**2)) / 4
    d = 4 * a * b / (1 + b) - (m - 1) * torch.log(torch.tensor(m - 1))
    Beta_dist = torch.distributions.Beta((m-1)/2, (m-1)/2)
    
    samples = []
    for i in range(batch_size): # for each sample
        while True:
            epsilon = Beta_dist.sample()
            omega = (1 - (1 + b[i]) * epsilon) / (1 - (1 - b[i]) * epsilon) # >> RAPH: Corrected parenthises of the numerator
            t = 2 * a[i] * b[i] / (1 - (1 - b[i]) * epsilon)
            u = torch.rand(1)
            if (m - 1) * torch.log(t) - t + d[i] >= torch.log(u):
                break
             
        # échantillonnage sur s^{m-2}
        v = torch.randn(m - 1)
        v = v / torch.norm(v)

        # construction du sample
        z = torch.cat([omega, 
                       torch.sqrt(1 - omega**2) * v])
        
        # transformation householder pour aligner avec mu
        e1 = torch.zeros(m)
        e1[0] = 1
        u = e1 - mu[i,:]
        u = u / (torch.norm(u) + 1e-8)
        
        householder = torch.eye(m) - 2 * torch.outer(u, u)
        z = householder @ z
        samples.append(z)
    
    return torch.stack(samples)

def sample_gaussian(mu, std):
    """
    Sampling from a gaussian
    """
    return mu + std * torch.randn_like(std)
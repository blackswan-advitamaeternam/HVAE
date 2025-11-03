import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

# clustering dans espace latent
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# test différents nombres de clusters
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(mu_tahoe_np)
    score = silhouette_score(mu_tahoe_np, clusters)
    silhouette_scores.append(score)

plt.figure(figsize=(6, 4))
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('nombre de clusters')
plt.ylabel('score silhouette')
plt.title('qualité clustering espace latent')
plt.grid(True, alpha=0.3)
plt.show()

best_k = k_range[np.argmax(silhouette_scores)]
print(f"nombre optimal de clusters: {best_k}")


# génération depuis prior uniforme sur sphère
def generate_from_uniform_sphere(model, n_samples=10, dim=10):
    # échantillonnage uniforme sur s^{d-1}
    z = torch.randn(n_samples, dim)
    z = z / torch.norm(z, dim=1, keepdim=True)
    
    with torch.no_grad():
        generated = model.decode(z)
    
    return generated

generated_samples = generate_from_uniform_sphere(model_tahoe, n_samples=5)

# interpolation sphérique entre deux points
def spherical_interpolation(z1, z2, n_steps=10):
    # normalisation
    z1 = z1 / torch.norm(z1)
    z2 = z2 / torch.norm(z2)
    
    # angle entre vecteurs
    omega = torch.acos(torch.clamp(torch.dot(z1, z2), -1, 1))
    
    interpolated = []
    for t in np.linspace(0, 1, n_steps):
        if omega > 1e-6:
            z_t = (torch.sin((1-t)*omega)/torch.sin(omega)) * z1 + (torch.sin(t*omega)/torch.sin(omega)) * z2
        else:
            z_t = (1-t) * z1 + t * z2
        interpolated.append(z_t)
    
    return torch.stack(interpolated)

# test interpolation
with torch.no_grad():
    idx1, idx2 = 0, 10
    z1 = mu_tahoe[idx1]
    z2 = mu_tahoe[idx2]
    
    z_interp = spherical_interpolation(z1, z2)
    x_interp = model_tahoe.decode(z_interp)

print(f"interpolation entre échantillons {idx1} et {idx2}")
print(f"forme interpolation: {x_interp.shape}")
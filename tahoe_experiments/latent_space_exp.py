import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np

from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import os, sys

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

# Custom imports
from tahoe_experiments.load_tahoe import prepare_tahoe_dataset
from tahoe_experiments.dim_reduction import preprocess

from svae.vae import SVAE, GaussianVAE
from svae.training import training
from svae.viz import save_plot

def plot_latent_spaces(svae_latent_tsne, 
                       svae_latent_umap,
                       nvae_latent_tsne,
                       nvae_latent_umap,
                       drug_colors=None,
                       save_path=None):
    cmap = 'tab20' if drug_colors is not None else None
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.scatter(svae_latent_tsne[:, 0], 
                svae_latent_tsne[:, 1], 
                c=drug_colors, cmap=cmap, 
                alpha=0.6, s=10)
    plt.title('espace latent S-VAE sur tahoe (projection t-sne)')
    plt.colorbar(label='drug id')
    plt.axis('equal')

    plt.subplot(2, 2, 2)
    plt.scatter(svae_latent_umap[:, 0], 
                svae_latent_umap[:, 1], 
                c=drug_colors, cmap=cmap, 
                alpha=0.6, s=10)
    plt.title('espace latent S-VAE sur tahoe (projection UMAP)')
    plt.colorbar(label='drug id')
    plt.axis('equal')

    plt.subplot(2, 2, 3)
    plt.scatter(nvae_latent_tsne[:, 0], 
                nvae_latent_tsne[:, 1], 
                c=drug_colors, cmap=cmap, 
                alpha=0.6, s=10)
    plt.title('espace latent N-VAE sur tahoe (projection t-sne)')
    plt.colorbar(label='drug id')
    plt.axis('equal')

    plt.subplot(2, 2, 4)
    plt.scatter(nvae_latent_umap[:, 0], 
                nvae_latent_umap[:, 1], 
                c=drug_colors, cmap=cmap, 
                alpha=0.6, s=10)
    plt.title('espace latent N-VAE sur tahoe (projection UMAP)')
    plt.colorbar(label='drug id')
    plt.axis('equal')

    plt.tight_layout()
    if save_path is not None:
        save_plot(save_path)
    else:
        plt.show()

def kappa_analysis(kappas, save_path):
    # analyse concentration kappa
    kappa_values = kappas.numpy().flatten()

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.hist(kappa_values, bins=30, alpha=0.7)
    plt.title('distribution des kappa')
    plt.xlabel('kappa')

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(kappa_values)), kappa_values, alpha=0.3, s=1)
    plt.title('kappa par échantillon')
    plt.xlabel('échantillon')
    plt.ylabel('kappa')

    msg = f"kappa moyen: {kappa_values.mean():.3f} +/- {kappa_values.std():.3f}"
    plt.suptitle(msg)

    plt.tight_layout()
    if save_path is not None:
        save_plot(save_path)
    else:
        plt.show()

def reconstruction_metrics(model, save_path):
    # métriques de reconstruction
    model.eval()
    with torch.no_grad():
        x_recon_tahoe, _, _ = model(tahoe_tensor)
        
        mse = F.mse_loss(x_recon_tahoe, tahoe_tensor).item()
        
        # corrélation par échantillon
        correlations = []
        for i in range(len(tahoe_tensor)):
            corr = np.corrcoef(tahoe_tensor[i].numpy(), 
                               x_recon_tahoe[i].numpy())[0, 1]
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)

    print(f"\nMSE reconstruction: {mse:.4f}")
    print(f"corrélation moyenne: {mean_corr:.4f}")

    # distribution des corrélations
    plt.figure(figsize=(6, 4))
    plt.hist(correlations, bins=30, alpha=0.7)
    plt.axvline(mean_corr, color='red', linestyle='--', 
                label=f'moyenne: {mean_corr:.3f}')
    plt.title('corrélations reconstruction')
    plt.xlabel('corrélation')
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        save_plot(save_path)
    else:
        plt.show()

# génération depuis prior uniforme sur sphère
def generate_from_uniform_sphere(model, n_samples=10, dim=10):
    # échantillonnage uniforme sur s^{d-1}
    z = torch.randn(n_samples, dim)
    z = z / torch.norm(z, dim=1, keepdim=True)
    
    model.eval()
    with torch.no_grad():
        generated = model.decode(z)
    
    return generated

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

def test_interpolation(mu, kappa, model):
    samples = model.sample(mu, kappa)
    
    # test interpolation
    model.eval()
    with torch.no_grad():
        idx1, idx2 = 0, 10
        z1 = samples[idx1]
        z2 = samples[idx2]
        
        z_interp = spherical_interpolation(z1, z2)
        x_interp = model.decode(z_interp)

    print(f"interpolation entre échantillons {idx1} et {idx2}")
    print(f"forme interpolation: {x_interp.shape}")
    return z_interp, x_interp

if __name__ == "__main__":
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    # load and prepare tahoe
    SIZE = 1000
    TRAIN_SIZE = int(0.8 * SIZE)
    n_components = 50
    data, labels = prepare_tahoe_dataset(SIZE)
    data_pca, pca = preprocess(data, n_components=n_components)

    sample_data_pca = data_pca[:TRAIN_SIZE,:]
    sample_labels = labels[:TRAIN_SIZE]
    val_sample_data_pca = data_pca[TRAIN_SIZE:,:]
    val_sample_labels = labels[TRAIN_SIZE:]

    # TRAIN: conversion torch
    tahoe_tensor = torch.FloatTensor(sample_data_pca)
    tahoe_dataset = TensorDataset(tahoe_tensor)
    tahoe_loader = DataLoader(tahoe_dataset, batch_size=16, shuffle=True)

    # VALIDATION: conversion torch
    val_tahoe_tensor = torch.FloatTensor(val_sample_data_pca)
    val_tahoe_dataset = TensorDataset(val_tahoe_tensor)
    val_tahoe_loader = DataLoader(val_tahoe_dataset, batch_size=16, shuffle=True)

    EPOCHS = 10
    # ===============SVAE pour Tahoe=============
    print("[SVAE] Instantiating SVAE and optimizer..")
    
    # >> RAPH: From what I've seen, SVAE is much more sensitive to hyperparameters
    # on this dataset, especially, the beta_kl needs to be low
    # It can freeze during training => investigate why 
    # also look at the warming up of beta kl (start with very small and 
    # gradually increase, I saw that in another paper)
    # 8 = multiplier of 2 + close to the maximum surface area of the sphere (d=7)
    latent_dim = 8
    hidden_dim = 256
    
    model_svae = SVAE(50, hidden_dim, latent_dim)  # dimension latente 10 pour capturer complexité
    optimizer = torch.optim.Adam(model_svae.parameters(), lr=5e-4)

    # training
    print("[SVAE] Started training..")
    model_svae, svae_losses, all_svae_parts = training(tahoe_loader, 
                                       val_tahoe_loader,
                                        model_svae,
                                        optimizer,
                                        epochs=EPOCHS,
                                        beta_kl=1e-4,
                                        patience=10,
                                        show_loss_every=1)
    # analyse espace latent tahoe
    svae_latent_samples, mu_svae, kappas = model_svae.get_latent_samples(tahoe_tensor)

    # projection tsne pour visualisation (dimension 10 -> 2)
    print("[SVAE] TSNE projection..")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    svae_latent_tsne = tsne.fit_transform(svae_latent_samples)

    # projection UMAP pour visualisation (dimension 10 -> 2)
    print("[SVAE] UMAP projection..")
    umap = UMAP(n_components=2, random_state=1234)
    svae_latent_umap = umap.fit_transform(svae_latent_samples)

    print("[SVAE] Kappa analaysis..")
    kappa_analysis(kappas, "./Plots/SVAE_kappas.pdf")

    print("[SVAE] Plotting Reconstruction metrics..")
    reconstruction_metrics(model_svae, "./Plots/SVAE_recon.pdf")

    # ===============NVAE pour Tahoe=============
    print("\n[NVAE] Instantiating NVAE and optimizer..")
    model_nvae = GaussianVAE(50, hidden_dim, latent_dim)  # dimension latente 10 pour capturer complexité
    optimizer = torch.optim.Adam(model_nvae.parameters(), lr=0.0001)

    # training
    print("[NVAE] Started training..")
    model_nvae, nvae_losses, all_nvae_parts = training(tahoe_loader, 
                                       val_tahoe_loader,
                                        model_nvae,
                                        optimizer,
                                        epochs=EPOCHS,
                                        beta_kl=0.1,
                                        patience=5,
                                        show_loss_every=1)
    # analyse espace latent tahoe
    nvae_latent_samples, mu_nvae, std = model_nvae.get_latent_samples(tahoe_tensor)

    print("[NVAE] Plotting Reconstruction metrics..")
    reconstruction_metrics(model_nvae, "./Plots/NVAE_recon.pdf")

    # projection tsne pour visualisation (dimension 10 -> 2)
    print("[NVAE] TSNE projection..")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    nvae_latent_tsne = tsne.fit_transform(nvae_latent_samples)

    # projection UMAP pour visualisation (dimension 10 -> 2)
    print("[NVAE] UMAP projection..")
    umap = UMAP(n_components=2, random_state=1234)
    nvae_latent_umap = umap.fit_transform(nvae_latent_samples)

    # encodage couleurs pour drugs
    unique_drugs = list(set(sample_labels))
    drug_colors = [unique_drugs.index(d) for d in sample_labels]

    print("Plotting latent spaces..")
    plot_latent_spaces(svae_latent_tsne, 
                       svae_latent_umap,
                       nvae_latent_tsne,
                       nvae_latent_umap,
                       drug_colors, 
                       "./Plots/compare_SVAE_NVAE_Tahoe.pdf")

    print("Generation uniform sphere prior..")
    generated_samples_svae = generate_from_uniform_sphere(model_svae, n_samples=100, dim=latent_dim)
    # projection tsne pour visualisation (dimension 10 -> 2)
    print("[SVAE] TSNE projection on generated..")
    gen_svae_latent_tsne = tsne.fit_transform(generated_samples_svae)

    # projection UMAP pour visualisation (dimension 10 -> 2)
    print("[SVAE] UMAP projection on generated..")
    gen_svae_latent_umap = umap.fit_transform(generated_samples_svae)

    generated_samples_nvae = generate_from_uniform_sphere(model_nvae, n_samples=100, dim=latent_dim)
    # projection tsne pour visualisation (dimension 10 -> 2)
    print("[NVAE] TSNE projection on generated..")
    gen_nvae_latent_tsne = tsne.fit_transform(generated_samples_nvae)

    # projection UMAP pour visualisation (dimension 10 -> 2)
    print("[NVAE] UMAP projection on generated..")
    gen_nvae_latent_umap = umap.fit_transform(generated_samples_nvae)
    
    plot_latent_spaces(gen_svae_latent_tsne, 
                       gen_svae_latent_umap,
                       gen_nvae_latent_tsne,
                       gen_nvae_latent_umap, 
                       save_path="./Plots/Uniform_sphere_compare_SVAE_NVAE_Tahoe.pdf")
    
    print("[SVAE] Interpolation test..")
    test_interpolation(mu_svae, kappas, model_svae)

    print("[NVAE] Interpolation test..")
    test_interpolation(mu_nvae, std, model_nvae)
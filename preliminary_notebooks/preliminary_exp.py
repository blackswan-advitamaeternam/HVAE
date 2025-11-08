import torch
import numpy as np
import os, sys

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

# Custom imports
from svae.vae import SVAE, GaussianVAE
from svae.sampling import sample_vmf, sample_gaussian
from svae.utils import create_circle_training_data
from svae.viz import compare_latent_space
from svae.training import training

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    # entrainement sur données circulaires
    print("Creating synthetic dataset..")
    SIZE = 1000
    TRAIN_SIZE = int(0.8 * SIZE)
    dataset, val_dataset = create_circle_training_data(SIZE, batch_size=32, train_size=TRAIN_SIZE)
    dataloader, data_tensor, data_2d, labels = dataset
    val_dataloader, val_data_tensor, val_data_2d, val_labels = val_dataset

    print("[SVAE] Instantiating SVAE and optimizer..")
    model_svae = SVAE(input_dim=100, hidden_dim=128, latent_dim=2)
    optimizer = torch.optim.Adam(model_svae.parameters(), lr=0.001)

    print("[SVAE] Started training..")
    model_svae, svae_losses, all_svae_parts = training(dataloader, 
                                    val_dataloader,
                                    model_svae,
                                    optimizer,
                                    epochs=50,
                                    beta_kl=0.1,
                                    show_loss_every=1)
    
    # visualisation espace latent SVAE
    svae_latent_samples, _, _ = model_svae.get_latent_samples(data_tensor)

    print("\n[NVAE] Instantiating NVAE and optimizer..")
    model_nvae = GaussianVAE(input_dim=100, hidden_dim=128, latent_dim=2)
    optimizer = torch.optim.Adam(model_nvae.parameters(), lr=0.001)

    print("[NVAE] Started training..")
    model_nvae, nvae_losses, all_nvae_parts = training(dataloader, 
                                    val_dataloader,
                                    model_nvae,
                                    optimizer,
                                    epochs=50,
                                    beta_kl=0.1,
                                    show_loss_every=1)
    
    # visualisation espace latent NVAE
    nvae_latent_samples, _, _ = model_nvae.get_latent_samples(data_tensor)
    
    print("Plotting latent spaces..")
    compare_latent_space(data_2d, 
                         svae_latent_samples, 
                         nvae_latent_samples,
                         labels, 
                         svae_losses,
                         nvae_losses, 
                         save_path="Plots/Compare_circle_SVAE_NVAE.pdf")
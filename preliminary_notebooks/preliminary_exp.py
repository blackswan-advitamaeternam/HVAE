import torch
import numpy as np
import os, sys

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

# Custom imports
from svae.vae import SVAE, GaussianVAE
from svae.sampling import sample_vmf
from svae.utils import create_circle_training_data
from svae.viz import plot_latent_space
from svae.training import training_loop

if __name__ == "__main__":
    # ensures the working dir is that of the file
    path_to_file = os.path.abspath(__file__)
    os.chdir('/'.join(path_to_file.split('/')[:-1]))

    # entrainement sur données circulaires
    print("Creating synthetic dataset..")
    dataset = create_circle_training_data(1000, batch_size=32)
    dataloader, data_tensor, data_2d, labels = dataset

    print("Instantiating SVAE and optimizer..")
    model_svae = SVAE(input_dim=100, hidden_dim=128, latent_dim=2)
    optimizer = torch.optim.Adam(model_svae.parameters(), lr=0.001)

    print("Started training..")
    model_svae, losses = training_loop(dataloader, model_svae, optimizer, 
                                  epochs=50, beta_kl=0.1,
                                  show_loss_every=1)

    # visualisation espace latent
    with torch.no_grad():
        print("Encoding dataset..")
        mu_all, kappa_all = model_svae.encode(data_tensor)
        # mu_np = mu_all.numpy()

        print("Sampling from latent space..")
        latent_samples = []
        for i in range(data_tensor.size(0)):
            z = sample_vmf(mu_all[i:i+1,:], kappa_all[i:i+1,:], 1)
            latent_samples.append(z.numpy())
        latent_samples = np.concat(latent_samples, axis=0)

    print("Plotting latent spaces..")
    plot_latent_space(data_2d, latent_samples, labels, losses, 
                      save_path="SVAE_compare_true_learned.pdf")
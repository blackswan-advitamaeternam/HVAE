import torch
import numpy as np
import os, sys
from sklearn.metrics import accuracy_score

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

# Custom imports
from svae.vae import M1_M2
from svae.viz import plot_latent_classes
from svae.training import training_M1M2

from paper_experiments.load_MNIST import make_splits_loaders_MNIST

if __name__ == "__main__":
    # ensures the working dir is that of the file
    file_parent = "/".join(os.path.abspath(__file__).split("/")[:-1])
    os.chdir(file_parent)

    # entrainement sur données circulaires
    train_loader, val_loader, test_loader = make_splits_loaders_MNIST()

    # Subsample
    train_loader = [batch for i, batch in enumerate(train_loader) if i < 50]
    val_loader = [batch for i, batch in enumerate(val_loader) if i < 10]
    test_loader = [batch for i, batch in enumerate(test_loader) if i < 10]

    EPOCHS = 10
    INPUT_DIM = 784
    HIDDEN_DIM = 128
    LATENT_DIM1 = 10
    LATENT_DIM2 = 10
    N_CLUSTERS = 10
    LATENT_MODE = 'sample'
    PATIENCE = max(2,int(0.1*EPOCHS))
    ONE_LAYER = True
    LR = 0.001
    print("[SVAE] Instantiating SVAE and optimizer..")
    model_svae = M1_M2("svae",
                    "svae",
                    INPUT_DIM,
                    HIDDEN_DIM,
                    LATENT_DIM1,
                    LATENT_DIM2,
                    N_CLUSTERS,
                    ONE_LAYER,
                    )
    optimizer = torch.optim.Adam(model_svae.parameters(), lr=LR)

    print("[SVAE] Started training..")
    model_svae, svae_losses, all_svae_parts = training_M1M2(train_loader, 
                                    val_loader,
                                    model_svae,
                                    optimizer,
                                    epochs=EPOCHS,
                                    beta_kl=1,
                                    alpha=0.1,
                                    patience=PATIENCE,
                                    show_loss_every=1)
    

    print("\n[NVAE] Instantiating NVAE and optimizer..")
    model_nvae = M1_M2("normal",
                    "normal",
                    INPUT_DIM,
                    HIDDEN_DIM,
                    LATENT_DIM1,
                    LATENT_DIM2,
                    N_CLUSTERS,
                    ONE_LAYER,
                    )
    optimizer = torch.optim.Adam(model_nvae.parameters(), lr=LR)

    print("[NVAE] Started training..")
    model_nvae, nvae_losses, all_nvae_parts = training_M1M2(train_loader, 
                                    val_loader,
                                    model_nvae,
                                    optimizer,
                                    epochs=EPOCHS,
                                    beta_kl=0.1,
                                    alpha=0.1,
                                    patience=PATIENCE,
                                    show_loss_every=1)

    print("\n[SVAE] Predicting classes..")
    plot_latent_classes(model_svae, test_loader, LATENT_MODE, save_path="./Plots/SVAE_LatentClassif.pdf")

    print("\n[NVAE] Predicting classes..")
    plot_latent_classes(model_nvae, test_loader, LATENT_MODE, save_path="./Plots/NVAE_LatentClassif.pdf")
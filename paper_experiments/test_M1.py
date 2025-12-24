import torch
import numpy as np
import os, sys
from sklearn.metrics import accuracy_score

# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

# Custom imports
from svae.vae import M1, predict_classes_loader
from svae.sampling import sample_vmf, sample_gaussian
from svae.utils import create_circle_training_data
from svae.viz import compare_latent_space
from svae.training import training_M1

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
    LATENT_DIM = 10
    LATENT_MODE = 'sample'
    PATIENCE = max(2,int(0.1*EPOCHS))
    ONE_LAYER = False
    LR = 0.001
    print("[SVAE] Instantiating SVAE and optimizer..")
    model_svae = M1("svae",
                    INPUT_DIM,
                    HIDDEN_DIM,
                    LATENT_DIM,
                    one_layer=True,
                    )
    optimizer = torch.optim.Adam(model_svae.parameters(), lr=LR)

    print("[SVAE] Started training..")
    model_svae, svae_losses, all_svae_parts = training_M1(train_loader, 
                                    val_loader,
                                    model_svae,
                                    optimizer,
                                    epochs=EPOCHS,
                                    beta_kl=1,
                                    patience=PATIENCE,
                                    show_loss_every=1,
                                    mode=LATENT_MODE)
    

    print("\n[NVAE] Instantiating NVAE and optimizer..")
    model_nvae = M1("normal",
                    INPUT_DIM,
                    HIDDEN_DIM,
                    LATENT_DIM,
                    one_layer=True,
                    )
    optimizer = torch.optim.Adam(model_nvae.parameters(), lr=LR)

    print("[NVAE] Started training..")
    model_nvae, nvae_losses, all_nvae_parts = training_M1(train_loader, 
                                    val_loader,
                                    model_nvae,
                                    optimizer,
                                    epochs=EPOCHS,
                                    beta_kl=0.1,
                                    patience=PATIENCE,
                                    show_loss_every=1,
                                    mode=LATENT_MODE)

    print("\n[SVAE] Predicting classes..")
    Y, Y_hat = predict_classes_loader(model_svae, test_loader, LATENT_MODE)

    test_acc = accuracy_score(Y, Y_hat)
    print(f"[SVAE] Test accuracy: {test_acc*100:.2f}")

    print("\n[NVAE] Predicting classes..")
    Y, Y_hat = predict_classes_loader(model_nvae, test_loader, LATENT_MODE)

    test_acc = accuracy_score(Y, Y_hat)
    print(f"[NVAE] Test accuracy: {test_acc*100:.2f}")
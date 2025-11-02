"""Sub-module to define vizualisation functions"""
import matplotlib.pyplot as plt
from typing import NoReturn
import numpy as np
import os

def plot_latent_space(true_2d_latent: np.ndarray, 
                      learned_2d_latent: np.ndarray, 
                      labels: np.ndarray, 
                      losses: list,
                      save_path: str = None,
                      ) -> NoReturn:
    """
    Function to plot and compare true vs learned latent spaces in 2D.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(true_2d_latent[:, 0], true_2d_latent[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('données originales s1')
    plt.axis('equal')

    plt.subplot(1, 3, 2)
    plt.scatter(learned_2d_latent[:, 0], learned_2d_latent[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('espace latent s-vae')
    plt.axis('equal')

    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('loss')
    plt.xlabel('epoch')

    plt.tight_layout()
    if save_path is not None:
        if "/" in save_path:
            os.makedirs(save_path, exist_ok=True)
        plot_ext = os.path.splitext(save_path)[-1][1:]
        plt.savefig(save_path, 
                    bbox_inches='tight', 
                    format=plot_ext)
        print(f"\nPlot saved at: {save_path}.")
    else:
        plt.show()

def compare_latent_space(true_2d_latent: np.ndarray, 
                      svae_learned_2d_latent: np.ndarray, 
                      nvae_learned_2d_latent: np.ndarray,
                      labels: np.ndarray, 
                      svae_losses: list,
                      nvae_losses: list,
                      save_path: str = None,
                      ) -> NoReturn:
    """
    Function to plot and compare true vs learned latent spaces in 2D.
    """
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 3, 1)
    plt.scatter(true_2d_latent[:, 0], 
                true_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('données originales s1')
    plt.axis('equal')

    plt.subplot(2, 3, 2)
    plt.scatter(svae_learned_2d_latent[:, 0], 
                svae_learned_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('espace latent S-VAE')
    plt.axis('equal')

    plt.subplot(2, 3, 3)
    plt.plot(svae_losses)
    plt.title('S-VAE training loss')
    plt.xlabel('epoch')

    plt.subplot(2, 3, 4)
    plt.scatter(true_2d_latent[:, 0], 
                true_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('données originales s1')
    plt.axis('equal')

    plt.subplot(2, 3, 5)
    plt.scatter(nvae_learned_2d_latent[:, 0], 
                nvae_learned_2d_latent[:, 1], 
                c=labels, cmap='viridis', alpha=0.5)
    plt.title('espace latent N-VAE')
    plt.axis('equal')

    plt.subplot(2, 3, 6)
    plt.plot(nvae_losses)
    plt.title('N-VAE training loss')
    plt.xlabel('epoch')

    plt.tight_layout()
    if save_path is not None:
        if "/" in save_path:
            os.makedirs(save_path, exist_ok=True)
        plot_ext = os.path.splitext(save_path)[-1][1:]
        plt.savefig(save_path, 
                    bbox_inches='tight', 
                    format=plot_ext)
        print(f"\nPlot saved at: {save_path}.")
    else:
        plt.show()
"""Sub-module to define training protocols"""
import torch.nn.functional as F
import torch

# Custom imports
from svae.vae import SVAE, GaussianVAE

# Global variable
SUPPORTED_CLASSES = [SVAE, GaussianVAE]

class EarlyStopping(object):
    """
    Implementation of early stopping with best epoch tracking
    """
    def __init__(self, patience):
        self.patience = patience
        self.losses = []
        self.loss_count = 0
        self.patience_count = 0
        self.best_loss = torch.inf
        self.best_model = None
        self.best_loss_idx = 0
    
    def register(self, model, loss):
        # Keep only patience losses
        if len(self.losses) == self.patience:
            self.losses = self.losses[1:]

        # register the loss
        self.losses.append(loss)
        self.loss_count += 1

        # check if this is the best loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_idx = self.loss_count
            self.best_model = model
            # reset the patience count
            self.patience_count = 0
        else:
            self.patience_count += 1
    
    def check_stop(self, model, loss):
        self.register(model, loss)
        if self.patience_count >= self.patience:
            return True
        return False
    
def training_svae(dataloader: torch.utils.data.DataLoader,
                  model_svae: SVAE, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 0.1,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  patience: int | None = 5,
                  show_loss_every: int = 10):
    """
    Training protocol for SVAE models.
    """
    assert isinstance(model_svae, SVAE), f"This training loop is tailored for SVAE modules"
    # instantiate early stopper
    early_stopper = EarlyStopping(patience) if patience is not None else None

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            x_recon, mu, kappa = model_svae(x)
            
            # reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)
            
            # kl divergence
            kl_loss = model_svae.kl_vmf(kappa)

            loss = recon_loss + beta_kl * kl_loss
            loss.backward()
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(dataloader)

        # check early stoppage
        if early_stopper.check_stop(model_svae, epoch_loss):
            print(f"\nEarly stoppage after {epoch} epochs with patience of {patience}.")
            final_loss_idx = early_stopper.best_loss_idx
            print(f"Best epoch: {final_loss_idx}")
            return early_stopper.best_model, losses[:final_loss_idx]
        
        losses.append(epoch_loss)
        if epoch % show_loss_every == 0:
            print(f"epoch {epoch}: {losses[-1]:.4f}")
    return model_svae, losses


def training_nvae(dataloader: torch.utils.data.DataLoader,
                  model_nvae: GaussianVAE, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 0.001,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  patience: int | None = 5,
                  show_loss_every: int = 10):
    """
    Training protocol for NVAE models.
    """
    assert isinstance(model_nvae, GaussianVAE), f"This training loop is tailored for SVAE modules"
    early_stopper = EarlyStopping(patience) if patience is not None else None

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model_nvae(x)
            
            recon_loss = F.mse_loss(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + beta_kl * kl_loss
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        epoch_loss = epoch_loss / len(dataloader)
        # check early stoppage
        if early_stopper.check_stop(model_nvae, epoch_loss):
            print(f"\nEarly stoppage after {epoch} epochs with patience of {patience}.")
            final_loss_idx = early_stopper.best_loss_idx
            print(f"Best epoch: {final_loss_idx}")
            return early_stopper.best_model, losses[:final_loss_idx]
        
        losses.append(epoch_loss)
        if epoch % show_loss_every == 0:
            print(f"epoch {epoch}: {losses[-1]:.4f}")
    return model_nvae, losses

def training_loop(dataloader: torch.utils.data.DataLoader,
                  model: SVAE | GaussianVAE, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 0.001,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  **kwargs):
    if isinstance(model, SVAE):
        return training_svae(dataloader,
                            model, 
                            optimizer, 
                            epochs,
                            beta_kl,
                            scheduler,
                            **kwargs)
    elif isinstance(model, GaussianVAE):
        return training_nvae(dataloader,
                            model, 
                            optimizer, 
                            epochs,
                            beta_kl,
                            scheduler,
                            **kwargs)
    else:
        raise ValueError(f"Unsupported model class: Only supports {SUPPORTED_CLASSES} but got {model.__class__}")
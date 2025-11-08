"""Sub-module to define training protocols"""
import torch.nn.functional as F
import torch
import numpy as np

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

def format_loss(val_epoch_parts, beta_kl):
    """
    print the loss components in the form:
    -recon - beta_kl * kl
    while handling signs cleanly (no '--' or '+ -').
    """
    recon = val_epoch_parts["recon"][-1]
    kl_latent = val_epoch_parts["kl"][-1]
    return f"- {recon:.4f} - {beta_kl} * {kl_latent:.4f}"

def training(dataloader: torch.utils.data.DataLoader,
                  val_dataloader:torch.utils.data.DataLoader,
                  model: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 1,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  patience: int | None = 10,
                  show_loss_every: int = 10):
    """
    Training protocol for our VAE models.
    """
    assert np.sum([isinstance(model, CLS) for CLS in SUPPORTED_CLASSES]) != 0, f"Unsupported model class: Only supports {SUPPORTED_CLASSES} but got {model.__class__}"
    # instantiate early stopper
    early_stopper = EarlyStopping(patience) if patience is not None else None

    losses = {"train":[], "val":[]}
    all_parts = {"train":{"recon":[],
                        "kl":[]},
                "val":{"recon":[],
                        "kl":[]}}
    for epoch in range(1,epochs+1):
        # TRAINING
        model.train()
        epoch_loss = 0
        epoch_parts = {"recon":[],
                       "kl":[]}
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()

            loss, parts = model.full_step(x, 
                                        beta_kl=beta_kl)
            
            loss.backward()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            for key in epoch_parts.keys():
                epoch_parts[key].append(parts[key])
            
        # register average epoch loss
        epoch_loss = epoch_loss / len(dataloader)
        # register average of epoch parts
        for key in epoch_parts.keys():
            epoch_parts[key].append(np.mean(epoch_parts[key]))

        # VALIDATION
        model.eval()
        val_epoch_loss = 0
        val_epoch_parts = {"recon":[],
                            "kl":[]}
        
        with torch.no_grad():
            for batch in val_dataloader:
                x = batch[0]
                loss, parts = model.full_step(x, 
                                            beta_kl=beta_kl)
                val_epoch_loss += loss.item()
                for key in val_epoch_parts.keys():
                    val_epoch_parts[key].append(parts[key])
                
            # register average epoch loss
            val_epoch_loss = val_epoch_loss / len(val_dataloader)
            # register average of epoch parts
            for key in val_epoch_parts.keys():
                val_epoch_parts[key].append(np.mean(val_epoch_parts[key]))
        
        # check early stoppage
        if early_stopper.check_stop(model, val_epoch_loss):
            print(f"\nEarly stoppage after {epoch} epochs with patience of {patience}.")
            final_loss_idx = early_stopper.best_loss_idx
            print(f"Best epoch: {final_loss_idx}")
            if final_loss_idx == 1:
                # 2 losses if the best epoch was the first
                # this avoids plotting a single point in other functions
                final_loss_idx = 2
            losses["train"] = losses["train"][:final_loss_idx]
            losses["val"] = losses["val"][:final_loss_idx]
            all_parts["train"] = {key:val[:final_loss_idx] for key, val in all_parts["train"].items()}
            all_parts["val"] = {key:val[:final_loss_idx] for key, val in all_parts["val"].items()}
            return early_stopper.best_model, losses, all_parts
        
        losses["train"].append(epoch_loss)
        losses["val"].append(val_epoch_loss)
        for key in all_parts["train"].keys():
            all_parts["train"][key].append(np.mean(epoch_parts[key]))
        for key in all_parts["val"].keys():
            all_parts["val"][key].append(np.mean(val_epoch_parts[key]))
        if epoch == 1:
            print("Loss printing format:\nepoch x: val = loss (-recon - beta_kl * kl) | train = loss (-recon - beta_kl * kl)\n")
        if epoch % show_loss_every == 0:
            print(f"epoch {epoch}: val = {losses["val"][-1]:.4f} ({format_loss(val_epoch_parts, beta_kl)}) | train = {losses["train"][-1]:.4f} ({format_loss(epoch_parts, beta_kl)})")
    return model, losses, all_parts
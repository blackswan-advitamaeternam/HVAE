"""Sub-module to define training protocols"""
import torch.nn.functional as F
import torch
import numpy as np
import copy
from tqdm.auto import tqdm

# Custom imports
from svae.vae import SVAE, GaussianVAE, M1, M1_M2, cluster_acc, predict_classes_loader

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
        self.best_state = None
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
            self.best_state = copy.deepcopy(model.state_dict())
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
    recon + beta_kl * kl
    while handling signs cleanly (no '--' or '+ -').
    """
    recon = val_epoch_parts["recon"][-1]
    kl_latent = val_epoch_parts["kl"][-1]
    return f"{recon:.4f} + {beta_kl:.2f} * {kl_latent:.4f}"

def format_loss_M1M2(val_epoch_parts):
    """
    print the loss components in the form:
    recon + beta_kl * kl
    while handling signs cleanly (no '--' or '+ -').
    """
    M1 = val_epoch_parts["M1"][-1]
    M2 = val_epoch_parts["M2"][-1]
    return f"{M1:.4f} + {M2:.4f}"

def to_item(dico):
    for key in dico.keys():
        val = dico[key]
        try:
            for i, el in enumerate(val):
                val[i] = el.item()
        except:
            print("Failed, skipping..")
            pass
    return dico

def training(dataloader: torch.utils.data.DataLoader,
                  val_dataloader:torch.utils.data.DataLoader,
                  model: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 1,
                  warmup=None,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  patience: int | None = 10,
                  show_loss_every: int = 10):
    """
    Training protocol for our VAE models.
    """
    assert np.sum([isinstance(model, CLS) for CLS in SUPPORTED_CLASSES]) != 0, f"Unsupported model class: Only supports {SUPPORTED_CLASSES} but got {model.__class__}"
    device = model.device

    if warmup is not None:
        beta_arr = np.linspace(0, beta_kl, warmup)
    else:
        beta_arr = None
        
    # instantiate early stopper
    early_stopper = EarlyStopping(patience) if patience is not None else None

    losses = {"train":[], "val":[]}
    all_parts = {"train":{"recon":[],
                        "kl":[]},
                "val":{"recon":[],
                        "kl":[]}}
    for epoch in tqdm(list(range(1,epochs+1)), desc="Epochs.."):
        # TRAINING
        model.train()
        epoch_loss = 0
        epoch_parts = {"recon":[],
                       "kl":[]}
        for batch in dataloader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            if beta_arr is not None and epoch <= warmup:
                beta_kl = beta_arr[epoch-1]
            loss, parts = model.full_step(x, 
                                        beta_kl=beta_kl)
            
            loss.backward()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            epoch_loss = epoch_loss + loss.detach()
            for key in epoch_parts.keys():
                epoch_parts[key].append(parts[key])
            
        # register average epoch loss
        epoch_loss = epoch_loss.item() / len(dataloader)
        # from tensor to float
        epoch_parts = to_item(epoch_parts)
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
                x = batch[0].to(device, non_blocking=True)
                loss, parts = model.full_step(x, 
                                            beta_kl=beta_kl)
                val_epoch_loss = val_epoch_loss + loss
                for key in val_epoch_parts.keys():
                    val_epoch_parts[key].append(parts[key])
                
            # register average epoch loss
            val_epoch_loss = val_epoch_loss.item() / len(val_dataloader)
            # from tensor to float
            val_epoch_parts = to_item(val_epoch_parts)
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
            model.load_state_dict(early_stopper.best_state)
            return model, losses, all_parts
        
        losses["train"].append(epoch_loss)
        losses["val"].append(val_epoch_loss)
        for key in all_parts["train"].keys():
            all_parts["train"][key].append(np.mean(epoch_parts[key]))
        for key in all_parts["val"].keys():
            all_parts["val"][key].append(np.mean(val_epoch_parts[key]))
        if epoch == 1:
            print("Loss printing format:\nepoch x: val = loss (-recon + beta_kl * kl) | train = loss (-recon + beta_kl * kl)\n")
        if epoch % show_loss_every == 0:
            print(f"epoch {epoch}: val = {losses["val"][-1]:.4f} ({format_loss(val_epoch_parts, beta_kl)}) | train = {losses["train"][-1]:.4f} ({format_loss(epoch_parts, beta_kl)})")
    return model, losses, all_parts


def training_M1(dataloader: torch.utils.data.DataLoader,
                  val_dataloader:torch.utils.data.DataLoader,
                  model: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer, 
                  epochs: int = 50,
                  beta_kl : float = 1,
                  warmup=None,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  patience: int | None = 10,
                  show_loss_every: int = 10,
                  mode="sample"):
    """
    Training protocol for our VAE models.
    """
    assert isinstance(model, M1), f"Unsupported model class: Only supports {M1} but got {model.__class__}"
    device = model.device

    if warmup is not None:
        beta_arr = np.linspace(0, beta_kl, warmup)
    else:
        beta_arr = None

    # instantiate early stopper
    early_stopper = EarlyStopping(patience) if patience is not None else None

    losses = {"train":[], "val":[]}
    all_parts = {"train":{"recon":[],
                        "kl":[]},
                "val":{"recon":[],
                        "kl":[]}}
    for epoch in tqdm(list(range(1,epochs+1)), desc="Epochs.."):
        # TRAINING
        model.train()
        epoch_loss = 0
        epoch_parts = {"recon":[],
                       "kl":[]}
        for batch in dataloader:
            x = batch[0].to(device, non_blocking=True)
            optimizer.zero_grad()

            if beta_arr is not None and epoch <= warmup:
                beta_kl = beta_arr[epoch-1]
            loss, parts = model.full_step(x, 
                                        beta_kl=beta_kl)
            
            loss.backward()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            epoch_loss = epoch_loss + loss.detach()
            for key in epoch_parts.keys():
                epoch_parts[key].append(parts[key])
            
        # register average epoch loss
        epoch_loss = epoch_loss.item() / len(dataloader)
        # from tensor to float
        epoch_parts = to_item(epoch_parts)
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
                x = batch[0].to(device, non_blocking=True)
                loss, parts = model.full_step(x, 
                                            beta_kl=beta_kl)
                val_epoch_loss = val_epoch_loss + loss
                for key in val_epoch_parts.keys():
                    val_epoch_parts[key].append(parts[key])
                
            # register average epoch loss
            val_epoch_loss = val_epoch_loss.item() / len(val_dataloader)
            # from tensor to float
            val_epoch_parts = to_item(val_epoch_parts)
            # register average of epoch parts
            for key in val_epoch_parts.keys():
                val_epoch_parts[key].append(np.mean(val_epoch_parts[key]))
        
        # check early stoppage
        if early_stopper.check_stop(model, val_epoch_loss):
            print(f"\nEarly stoppage after {epoch} epochs with patience of {patience}.")
            print(f"Fitting KNN classifier..")
            # fit KNN
            X_Y_pairs = [(batch[0].to(device, non_blocking=True),batch[1].to(device, non_blocking=True)) for batch in dataloader]
            model.fit_clf(torch.cat([pair[0] for pair in X_Y_pairs]), 
                          torch.cat([pair[1] for pair in X_Y_pairs]), 
                          mode)
            
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
            model.load_state_dict(early_stopper.best_state)
            return model, losses, all_parts
        
        losses["train"].append(epoch_loss)
        losses["val"].append(val_epoch_loss)
        for key in all_parts["train"].keys():
            all_parts["train"][key].append(np.mean(epoch_parts[key]))
        for key in all_parts["val"].keys():
            all_parts["val"][key].append(np.mean(val_epoch_parts[key]))
        if epoch == 1:
            print("Loss printing format:\nepoch x: val = loss (-recon + beta_kl * kl) | train = loss (-recon + beta_kl * kl)\n")
        if epoch % show_loss_every == 0:
            print(f"epoch {epoch}: val = {losses["val"][-1]:.4f} ({format_loss(val_epoch_parts, beta_kl)}) | train = {losses["train"][-1]:.4f} ({format_loss(epoch_parts, beta_kl)})")
    
    print(f"Fitting KNN classifier..")
    # fit KNN
    X_Y_pairs = [(batch[0].to(device, non_blocking=True),batch[1].to(device, non_blocking=True)) for batch in dataloader]
    model.fit_clf(torch.cat([pair[0] for pair in X_Y_pairs]), 
                    torch.cat([pair[1] for pair in X_Y_pairs]), 
                    mode)
    model.load_state_dict(early_stopper.best_state)
    return model, losses, all_parts

def training_M1M2(
    unlabeled_dataloader: torch.utils.data.DataLoader,
    labeled_dataloader: torch.utils.data.DataLoader,
    unlabeled_val_dataloader: torch.utils.data.DataLoader,
    labeled_val_dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 50,
    beta_kl: float = 1.0,
    warmup: int | None = None,
    alpha: float = 0.1,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    patience: int | None = 10,
    show_loss_every: int = 10,
    track_val_acc: bool = True,
):
    """
    Semi-supervised M1+M2 training.
    
    MODES (Mutually Exclusive for Validation):
      - track_val_acc=True : validation computes ONLY accuracy (Hungarian). 
                             Early stopping monitors -accuracy.
      - track_val_acc=False: validation computes ONLY ELBO loss. 
                             Early stopping monitors val_loss.
    """
    assert isinstance(model, M1_M2), (
        f"Unsupported model class: Only supports M1_M2 but got {model.__class__}"
    )
    device = model.device
    model.move_cat_dist()

    # KL warmup schedule
    beta_sched = None
    if warmup is not None and warmup > 0:
        beta_sched = np.linspace(0.0, float(beta_kl), warmup)

    early_stopper = EarlyStopping(patience) if patience is not None else None

    # Initialize metric storage based on mode
    losses = {"train": [], "val": []}
    
    if track_val_acc:
        # Only tracking accuracy for validation
        all_parts = {
            "train": {"M1": [], "M2": []},
            "val": {"acc": []} 
        }
    else:
        # Only tracking loss components for validation
        all_parts = {
            "train": {"M1": [], "M2": []},
            "val": {"M1": [], "M2": []}
        }

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs.."):
        # =========================
        #         TRAIN
        # =========================
        model.train()

        if beta_sched is not None and epoch <= warmup:
            beta_now = float(beta_sched[epoch - 1])
        else:
            beta_now = float(beta_kl)

        train_loss_sum = 0.0
        train_steps = 0
        train_parts_M1 = []
        train_parts_M2 = []

        # Fix: Cycle labeled data so we don't stop early
        for unlabeled_batch, labeled_batch in zip(unlabeled_dataloader, labeled_dataloader):
            x_u = unlabeled_batch[0].to(device, non_blocking=True)
            
            x_l = labeled_batch[0].to(device, non_blocking=True)
            y_l = labeled_batch[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Unlabeled Step
            loss_u, parts_u = model.full_step(
                x_u, None, beta_kl=beta_now, alpha=None, mode="unlabeled"
            )
            # Labeled Step
            loss_l, parts_l = model.full_step(
                x_l, y_l.long(), beta_kl=beta_now, alpha=alpha, mode="labeled"
            )

            # Total loss
            loss = loss_u + loss_l
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            train_loss_sum += float(loss.detach().item())
            train_steps += 1

            # Log parts safely
            m1_val = (parts_u["M1"] + parts_l["M1"]).detach().item() if torch.is_tensor(parts_u["M1"]) else parts_u["M1"] + parts_l["M1"]
            m2_val = (parts_u["M2"] + parts_l["M2"]).detach().item() if torch.is_tensor(parts_u["M2"]) else parts_u["M2"] + parts_l["M2"]
            
            train_parts_M1.append(float(m1_val))
            train_parts_M2.append(float(m2_val))

        epoch_train_loss = train_loss_sum / max(train_steps, 1)
        epoch_train_parts = {
            "M1": float(np.mean(train_parts_M1)) if train_parts_M1 else 0.0,
            "M2": float(np.mean(train_parts_M2)) if train_parts_M2 else 0.0,
        }

        # =========================
        #       VALIDATION
        # =========================
        model.eval()
        
        # --- BRANCH 1: ACCURACY TRACKING ---
        if track_val_acc:
            # Skip ELBO computation completely
            # Use 'sample' mode for accuracy as standard for M2
            # Assuming predict_classes_loader handles model.eval() internally or we set it above
            Y_true, Y_pred = predict_classes_loader(model, labeled_val_dataloader, mode="sample", return_latent=False)
            
            epoch_val_acc = cluster_acc(Y_true, Y_pred)
            
            # Store Metrics
            all_parts["val"]["acc"].append(epoch_val_acc)
            losses["val"].append(0.0) # Placeholder to keep length consistent
            
            monitor_metric = -epoch_val_acc # Minimize negative accuracy
            log_val_str = f"acc = {epoch_val_acc*100:.2f}%"

        # --- BRANCH 2: LOSS TRACKING ---
        else:
            val_loss_sum = 0.0
            val_steps = 0
            val_parts_M1 = []
            val_parts_M2 = []

            with torch.no_grad():
                for unlabeled_batch, labeled_batch in zip(unlabeled_val_dataloader, labeled_val_dataloader):
                    x_u = unlabeled_batch[0].to(device, non_blocking=True)
                    x_l = labeled_batch[0].to(device, non_blocking=True)
                    y_l = labeled_batch[1].to(device, non_blocking=True)

                    loss_u, parts_u = model.full_step(
                        x_u, None, beta_kl=beta_now, alpha=None, mode="unlabeled"
                    )
                    loss_l, parts_l = model.full_step(
                        x_l, y_l.long(), beta_kl=beta_now, alpha=alpha, mode="labeled"
                    )

                    loss = loss_u + loss_l
                    val_loss_sum += float(loss.detach().item())
                    val_steps += 1

                    m1_val = (parts_u["M1"] + parts_l["M1"]).detach().item() if torch.is_tensor(parts_u["M1"]) else parts_u["M1"] + parts_l["M1"]
                    m2_val = (parts_u["M2"] + parts_l["M2"]).detach().item() if torch.is_tensor(parts_u["M2"]) else parts_u["M2"] + parts_l["M2"]
                    val_parts_M1.append(float(m1_val))
                    val_parts_M2.append(float(m2_val))

            epoch_val_loss = val_loss_sum / max(val_steps, 1)
            
            # Store Metrics
            losses["val"].append(epoch_val_loss)
            all_parts["val"]["M1"].append(float(np.mean(val_parts_M1)) if val_parts_M1 else 0.0)
            all_parts["val"]["M2"].append(float(np.mean(val_parts_M2)) if val_parts_M2 else 0.0)
            
            monitor_metric = epoch_val_loss # Minimize loss
            log_val_str = f"loss = {epoch_val_loss:.4f}"

        # =========================
        #     LOGGING & STOPPING
        # =========================
        losses["train"].append(epoch_train_loss)
        all_parts["train"]["M1"].append(epoch_train_parts["M1"])
        all_parts["train"]["M2"].append(epoch_train_parts["M2"])

        if show_loss_every > 0 and epoch % show_loss_every == 0:
            print(f"Epoch {epoch:3d} | Train: {epoch_train_loss:.4f} | Val: {log_val_str}")

        if early_stopper is not None and early_stopper.check_stop(model, monitor_metric):
            print(f"\nEarly stoppage after {epoch} epochs.")
            best_idx = early_stopper.best_loss_idx # 0-based index of best epoch
            print(f"Best epoch was {best_idx + 1}") # Print as 1-based

            model.load_state_dict(early_stopper.best_state)
            
            # Trim history
            final_len = max(best_idx + 1, 1)
            losses["train"] = losses["train"][:final_len]
            losses["val"] = losses["val"][:final_len]
            
            for k in all_parts["train"]:
                all_parts["train"][k] = all_parts["train"][k][:final_len]
            
            if track_val_acc:
                 all_parts["val"]["acc"] = all_parts["val"]["acc"][:final_len]
            else:
                 for k in all_parts["val"]:
                    all_parts["val"][k] = all_parts["val"][k][:final_len]
            
            return model, losses, all_parts

    # Restore best model if finished without early stop
    if early_stopper is not None:
        model.load_state_dict(early_stopper.best_state)

    return model, losses, all_parts
"""Sub-module to define diverse utility functions"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.special
from numbers import Number
from typing import Tuple, List

# =========================
# computation utils
# =========================
# The following section was taken from the official implementation

class Ive(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z):

        assert isinstance(v, Number), "v must be a scalar"

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:  #  v > 0
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        #         else:
        #             print(v, type(v), np.isclose(v, 0))
        #             raise RuntimeError('v must be >= 0, it is {}'.format(v))

        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return (
            None,
            grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z),
        )

ive = Ive.apply

# class Ive(torch.nn.Module):
#     def __init__(self, v):
#         super(Ive, self).__init__()
#         self.v = v

#     def forward(self, z):
#         return ive(self.v, z)



##########
# The below provided approximations were provided in the
# respective source papers, to improve the stability of
# the Bessel fractions.
# I_(v/2)(k) / I_(v/2 - 1)(k)

# source: https://arxiv.org/pdf/1606.02008.pdf
def ive_fraction_approx(v, z):
    # I_(v/2)(k) / I_(v/2 - 1)(k) >= z / (v-1 + ((v+1)^2 + z^2)^0.5
    return z / (v - 1 + torch.pow(torch.pow(v + 1, 2) + torch.pow(z, 2), 0.5))


# source: https://arxiv.org/pdf/1902.02603.pdf
def ive_fraction_approx2(v, z, eps=1e-20):
    def delta_a(a):
        lamb = v + (a - 1.0) / 2.0
        return (v - 0.5) + lamb / (
            2 * torch.sqrt((torch.pow(lamb, 2) + torch.pow(z, 2)).clamp(eps))
        )

    delta_0 = delta_a(0.0)
    delta_2 = delta_a(2.0)
    B_0 = z / (
        delta_0 + torch.sqrt((torch.pow(delta_0, 2) + torch.pow(z, 2))).clamp(eps)
    )
    B_2 = z / (
        delta_2 + torch.sqrt((torch.pow(delta_2, 2) + torch.pow(z, 2))).clamp(eps)
    )

    return (B_0 + B_2) / 2.0

# =============================
# Shuffling manifested loaders
# =============================

import math
import torch
from typing import Iterable, List, Sequence, Optional, Union, Iterator

class ShuffledLoader:
    """
    Wraps an in-memory list of batches (e.g., manifested GPU batches) and yields them
    in a reshuffled order each time you iterate.

    Notes:
      - Shuffles *batch order* by default.
      - Optionally shuffles *within each batch* (same permutation applied to all tensors in the batch)
        if tensors are indexable on dim 0.
      - The underlying data remains on whatever device it already lives on.
    """
    def __init__(
        self,
        batches: Sequence[Sequence[torch.Tensor]],
        *,
        shuffle_batches: bool = True,
        shuffle_within_batch: bool = False,
        generator: Optional[torch.Generator] = None,
        device_for_randperm: Optional[Union[str, torch.device]] = None,
        drop_last: bool = False,
    ):
        self.batches = list(batches)
        self.shuffle_batches = shuffle_batches
        self.shuffle_within_batch = shuffle_within_batch
        self.generator = generator
        self.device_for_randperm = device_for_randperm  # None => torch.randperm default device (CPU)
        self.drop_last = drop_last

        # Basic validation: each batch is a sequence of tensors
        if len(self.batches) == 0:
            raise ValueError("ShuffledLoader received empty batches.")
        for b in self.batches:
            if not isinstance(b, (list, tuple)) or len(b) == 0:
                raise ValueError("Each batch must be a non-empty list/tuple of tensors.")

    def __len__(self) -> int:
        return len(self.batches)

    def _maybe_shuffle_within(self, batch: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not self.shuffle_within_batch:
            return list(batch)

        # Determine batch size from first tensor dim-0
        first = batch[0]
        if not isinstance(first, torch.Tensor) or first.ndim == 0:
            return list(batch)

        n = first.shape[0]
        # Ensure all tensors share the same leading dimension n when applicable
        for t in batch:
            if isinstance(t, torch.Tensor) and t.ndim > 0 and t.shape[0] != n:
                # If shapes mismatch, don't attempt within-batch shuffle
                return list(batch)

        perm = torch.randperm(
            n,
            generator=self.generator,
            device=self.device_for_randperm
        )
        out = []
        for t in batch:
            if isinstance(t, torch.Tensor) and t.ndim > 0 and t.shape[0] == n:
                out.append(t.index_select(torch.tensor(0, device=self.device_for_randperm), perm))
            else:
                out.append(t)
        return out

    def __iter__(self) -> Iterator[List[torch.Tensor]]:
        idx = torch.arange(len(self.batches), 
                          device=self.device_for_randperm)
        if self.shuffle_batches:
            perm = torch.randperm(
                len(self.batches),
                generator=self.generator,
                device=self.device_for_randperm
            )
            idx = idx[perm]

        for i in idx.tolist():
            yield self._maybe_shuffle_within(self.batches[i])

    def set_epoch(self, epoch: int) -> None:
        """
        Optional: call this each epoch if you want deterministic-but-different shuffles.
        If a generator was provided, we reseed it based on epoch.
        """
        if self.generator is None:
            self.generator = torch.Generator()
        # A simple epoch-dependent seed; adjust to your reproducibility scheme as needed.
        self.generator.manual_seed(10_000 + int(epoch))

# =========================
# Checking utils
# =========================
def has_method(obj: object, method_name: str) -> bool:
    """
    Checks if an object has the desired method
    """
    return callable(getattr(obj, method_name, None))

# =========================
# Synthetic data generation
# =========================

# génération données sur s1 embedées dans r100
def generate_circle_data(n_samples=1000
                         ) -> Tuple[np.ndarray | List]:
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    
    # 3 clusters sur le cercle
    cluster_centers = [0, 2*np.pi/3, 4*np.pi/3]
    labels = np.random.choice(3, n_samples)
    
    for i in range(n_samples):
        angles[i] = cluster_centers[labels[i]] + np.random.randn() * 0.3
    
    # coordonnées 2d
    x = np.cos(angles)
    y = np.sin(angles)
    data_2d = np.stack([x, y], axis=1)
    
    # projection non-linéaire vers r100
    projection = np.random.randn(2, 100)
    data_high = data_2d @ projection
    data_high += np.random.randn(n_samples, 100) * 0.1  # bruit
    
    return data_high, data_2d, labels

def create_circle_training_data(n_samples, 
                                batch_size,
                                train_size,
                                ) -> Tuple:
    """
    Creates a synthetic circle dataset and instantiates the dataloader.
    """
    data_high, data_2d, labels = generate_circle_data(n_samples)
    data_high, val_data_high = data_high[:train_size,:], data_high[train_size:,:]
    data_2d, val_data_2d = data_2d[:train_size,:], data_2d[train_size:,:]
    labels, val_labels = labels[:train_size], labels[train_size:]

    # TRAIN
    data_tensor = torch.FloatTensor(data_high)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # VAL
    val_data_tensor = torch.FloatTensor(val_data_high)
    val_dataset = TensorDataset(val_data_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return ((dataloader, data_tensor, data_2d, labels),
            (val_dataloader, val_data_tensor, val_data_2d, val_labels))
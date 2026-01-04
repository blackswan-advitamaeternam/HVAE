<div align="center">

# Hyperspherical Variational Auto-Encoders

**A Complete Reproduction & Critical Analysis**

[![Paper](https://img.shields.io/badge/arXiv-1804.00891-b31b1b.svg)](https://arxiv.org/abs/1804.00891)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="https://img.shields.io/badge/MVA-Computational%20Statistics-darkgreen.svg" alt="MVA"/>

[Paper](https://arxiv.org/abs/1804.00891) • [Original Code](https://github.com/nicola-decao/s-vae-pytorch) • [Slides](./slides/presentation.pdf)

---

*Reproduction study for Prof. Stéphanie Allassonnière - Master MVA, ENS Paris-Saclay*

**Authors:** [Mouhssine Rifaki](https://github.com/blackswan-advitamaeternam) & [Raphaël Rubrice](https://github.com/raphaelrubrice)

</div>

---

## Overview

This repository provides a **from-scratch reimplementation** of the S-VAE (Spherical Variational Auto-Encoder) proposed by Davidson et al. (2018), which replaces the standard Gaussian latent space with a **von Mises-Fisher (vMF) distribution** on the hypersphere.

### Why does this matter?

| Problem with Gaussian VAE | S-VAE Solution |
|---------------------------|----------------|
| Low-dim: prior pulls all points to origin | Uniform prior on sphere - no "gravity" |
| High-dim: mass concentrates on thin shell | Natural geometry for directional data |
| KL vanishing with powerful decoders | Better latent utilization |

<details>
<summary><b> Key Technical Contributions of the Paper</b></summary>

1. **Learnable concentration** $\kappa$ (vs. fixed in prior work)
2. **Reparameterization trick** for vMF via acceptance-rejection (Lemma 2)
3. **Closed-form KL divergence** between vMF and uniform on $S^{m-1}$

$$\text{KL}(\text{vMF}(\mu, \kappa) \| \mathcal{U}(S^{m-1})) = \kappa \frac{I_{m/2}(\kappa)}{I_{m/2-1}(\kappa)} + \log C_m(\kappa) + \text{const}$$

</details>

---

## Quick Start

### Prerequisites

We use [**uv**](https://github.com/astral-sh/uv) - the blazingly fast Python package manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
# Clone
git clone https://github.com/blackswan-advitamaeternam/HVAE.git
cd HVAE

# Setup environment
uv venv
source .venv/bin/activate
uv sync

# Verify installation
python -c "from svae import SVAE, GaussianVAE; print('Ready')"
```

---

## Reproduction Results

We reproduced **4 of 5** core experiments from the paper. Each experiment is fully documented with runnable notebooks.

### Summary

| Experiment | Paper Reference | Status | Notebook |
|:-----------|:----------------|:------:|:--------:|
| Circular manifold recovery | Figure 1 | Y | [Script](./preliminary_notebooks/preliminary_exp.py) |
| Unsupervised MNIST metrics | Table 1 | Y | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blackswan-advitamaeternam/HVAE/blob/main/paper_experiments/Table1_exp.ipynb) |
| Semi-supervised M1 (K-NN) | Table 2 | Y | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blackswan-advitamaeternam/HVAE/blob/main/paper_experiments/Table2_exp.ipynb) |
| Semi-supervised M1+M2 | Table 3 | Y | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/blackswan-advitamaeternam/HVAE/blob/main/paper_experiments/Table3_exp.ipynb) |

---

### Exp 1: Recovering Circular Structure (Figure 1)

Data sampled from 3 vMF distributions on $S^1$, embedded in $\mathbb{R}^{100}$.

```bash
python preliminary_notebooks/preliminary_exp.py
```

<div align="center">

| Ground Truth | S-VAE Latent | N-VAE Latent |
|:------------:|:------------:|:------------:|
| <img src="./assets/original_s1.png" width="200"/> | <img src="./assets/svae_latent.png" width="200"/> | <img src="./assets/nvae_latent.png" width="200"/> |
| 3 clusters on circle | Structure preserved | Collapsed to origin |

</div>

**Result:** S-VAE achieves **+13 nats** better log-likelihood by respecting circular geometry.

---

### Exp 2: Unsupervised MNIST (Table 1)

Metrics: Log-likelihood (IWAE, 500 samples), ELBO, Reconstruction Error, KL divergence.

<div align="center">

| $d$ | Model | LL ↑ | RE ↑ | KL |
|:---:|:-----:|:----:|:----:|:--:|
| 2 | N-VAE | -135.73 | -129.84 | 7.24 |
| 2 | **S-VAE** | **-132.50** | **-126.43** | 7.28 |
| 5 | N-VAE | -110.21 | -100.16 | 12.82 |
| 5 | **S-VAE** | **-108.43** | **-97.84** | 13.35 |
| 10 | **S-VAE** | **-93.16** | **-77.03** | 20.67 |
| 20 | N-VAE | **-88.90** | -71.29 | 23.50 |

</div>

**Insight:** S-VAE dominates at low dimensions ($d \leq 10$), N-VAE catches up at higher $d$ where the hypersphere surface area vanishes.

---

### Exp 3 & 4: Semi-Supervised Classification (Tables 2 & 3)

K-NN classification accuracy on learned latent representations.

**Key Finding:** The hybrid **S+N architecture** (spherical M1 + Gaussian M2) achieves best results in **8/9 configurations**, confirming the paper's recommendation.

---

## Project Structure

```
HVAE/
├── svae/                       # Core library
│   ├── vae.py                  # SVAE, GaussianVAE, M1, M1_M2 models
│   ├── sampling.py             # vMF sampling (Ulrich 1984)
│   ├── training.py             # Training loops with early stopping
│   └── utils.py                # Bessel functions, numerical stability
├── paper_experiments/          # Reproduction notebooks
│   ├── Table1_exp.ipynb        # Unsupervised metrics
│   ├── Table2_exp.ipynb        # M1 semi-supervised
│   ├── Table3_exp.ipynb        # M1+M2 semi-supervised
│   └── load_MNIST.py           # Data loading utilities
├── preliminary_notebooks/      # Initial experiments
│   └── preliminary_exp.py      # Figure 1 reproduction
└── requirements.txt
```

---

## Implementation Highlights

<details>
<summary><b>vMF Sampling via Rejection (Ulrich 1984)</b></summary>

```python
# Sample w ~ g(w|κ,m) in 1D, then lift to sphere
def sample_vmf(mu, kappa, n_samples):
    # 1. Rejection sampling for w (scalar)
    w = rejection_sample_w(kappa, dim)
    
    # 2. Sample v uniformly on S^{m-2}
    v = sample_uniform_sphere(dim - 1)
    
    # 3. Construct z' = (w, sqrt(1-w²) * v)
    z_prime = concat(w, sqrt(1 - w**2) * v)
    
    # 4. Householder rotation to align with μ
    z = householder_rotation(z_prime, mu)
    return z
```

**Key insight:** Rejection happens in 1D only → no curse of dimensionality.

</details>

<details>
<summary><b>Numerically Stable Bessel Functions</b></summary>

We use the **exponentially scaled** modified Bessel function $I_v^e(\kappa) = e^{-\kappa} I_v(\kappa)$ to prevent overflow when $\kappa$ is large.

```python
# Custom autograd for backprop through Bessel ratio
class Ive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, z):
        ctx.save_for_backward(z)
        ctx.v = v
        return scipy.special.ive(v, z.cpu().numpy())
    
    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        return None, grad_output * (ive(ctx.v-1, z) - ive(ctx.v, z) * (ctx.v + z) / z)
```

</details>

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{davidson2018hyperspherical,
  title={Hyperspherical Variational Auto-Encoders},
  author={Davidson, Tim R. and Falorsi, Luca and De Cao, Nicola and Kipf, Thomas and Tomczak, Jakub M.},
  booktitle={34th Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2018}
}
```

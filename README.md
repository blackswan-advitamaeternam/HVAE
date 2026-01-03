# **Hyperspherical Variational Auto-Encoders**

Work on the [Hyperspherical Variational Auto-Encoders](https://arxiv.org/abs/1804.00891) paper by Tim R. Davidson, Luca Falorsi, Nicola De Cao et al.
The objective of this project is to prepare a presentation to Pr. Stéphanie ALLASSONNIERE that showcases a deep understanding of the paper:
- What is new, or at least claimed to be ?
- What is interesting in the approach ?
- What is not interesting ?

To do so, we decided to reimplement their approach, reproduce their experiments and even extend it on a different application: scRNA-seq data. Through these initatives we hope to display a very strong grasp of the authors' work.

## **Installation**
We will use the modern package management system for python called [uv](https://github.com/astral-sh/uv)

1) To install uv on your device follow instructions [here](https://github.com/astral-sh/uv)

2) Clone the repo
```bash
git clone https://github.com/blackswan-advitamaeternam/HVAE.git
```

3) Go to the current repo (HVAE) and create a virtual environment
```bash
cd HVAE/
uv venv
```

4) Activtae the virtual environment and add the repo's library 
```bash
source .venv/bin/activate
uv sync
```

## **Reimplementation**
We decided to code their approach using PyTorch and Scipy.

## **Reproduction of experiments**
#### **Recovering a 2D Manifold on the circle**
Experiment leading to observing the latent space learned using N-VAE compared to S-VAE as done in Fig. 1 of the paper.

```bash
python preliminary_notebooks/preliminary_exp.py
``` 

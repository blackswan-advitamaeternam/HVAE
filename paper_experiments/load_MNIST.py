import tensorflow_datasets as tfds
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import os
import pickle as pkl
from pathlib import Path
FILEPATH = Path(os.path.abspath(__file__))
print(FILEPATH)
PARENTFOLDER = FILEPATH.parent
print(PARENTFOLDER)

# Dynamically binarize the images
def binarize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # Dynamic binarization: sample from Bernoulli distribution
    image = tf.cast(tf.random.uniform(tf.shape(image)) < image, tf.float32)
    return image, label

def load_binarized_mnist_tensor(split="train", batch_size=128, device=None):
    # Load all examples of the given split as a single batch
    ds = tfds.load('mnist', shuffle_files=True, split=split, batch_size=batch_size, as_supervised=True)
    # apply dynamic binarization as in Salakhutdinov & Murray, 2008
    ds = ds.map(binarize)
    ds_np = tfds.as_numpy(ds)
    # ds_np["image"]: shape (N, 28, 28), values {0,1}
    print(len(ds_np))
    x = torch.from_numpy(np.concatenate([itm[0] for itm in ds_np])).float()
    x = torch.flatten(x, start_dim=1)
    print(x.size())
    y = torch.from_numpy(np.concatenate([itm[1] for itm in ds_np])).float()

    if device is not None:
        return x.to(device), y.to(device)
    return x, y

def load_binarized_mnist_torch(split="train", batch_size=128, shuffle=True, device=None):
    x, y = load_binarized_mnist_tensor(split, batch_size, device)

    if device is not None:
        dataset = TensorDataset(x.to(device), y.to(device))
    else:
        dataset = TensorDataset(x, y)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle and (split == "train"))
    return loader

def get_loaders_MNIST():
    """
    Outputs the train, val and test data loaders for Binarized MNIST (2008)
    """
    train_loader = load_binarized_mnist_torch("train")
    test_loader  = load_binarized_mnist_torch("test", shuffle=False)
    return train_loader, test_loader

def make_cv_loaders_MNIST(cv=5, batch_size=100, device=None, force=False, **kwargs):
    """
    Loads the dynamically binarized MNIST trainset, applies
    a stratified K Fold CV split on it to form (train_loader, val_loader)
    pairs. Then loads the testset.
    Returns a list of (train_loader, val_loader) pairs for CV and the test loader
    """
    if "cv_splitted_MNIST.pkl" not in os.listdir(str(PARENTFOLDER)) or force:
        print("\nMaking splits..")
        X, Y = load_binarized_mnist_tensor("train", batch_size=batch_size, device=device)
        skf = StratifiedKFold(n_splits=cv)
        split_indices = skf.split(X, Y)

        cv_list = []
        for i, (train_index, val_index) in enumerate(split_indices):
            train_X, train_Y = X[train_index,:], Y[train_index]
            val_X, val_Y = X[val_index,:].reshape(len(val_index),784), Y[val_index]

            train_dataset = TensorDataset(train_X, train_Y) 
            train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                **kwargs)
            
            val_dataset = TensorDataset(val_X, val_Y) 
            val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                **kwargs)
            cv_list.append((train_loader, val_loader))

        test_loader  = load_binarized_mnist_torch("test", batch_size=batch_size, shuffle=False, device=device)

        # save for future use
        path = PARENTFOLDER / "cv_splitted_MNIST.pkl"
        with open(str(path), "wb") as f:
            pkl.dump({"cv": cv_list, "test": test_loader}, f)
    else:
        print("\nLoading pre-computed splits..")
        path = PARENTFOLDER / "cv_splitted_MNIST.pkl"
        # load splits
        with open(str(path), "rb") as f:
            dico = pkl.load(f)
            cv_list, test_loader = dico["cv"], dico["test"]
    return cv_list, test_loader

def make_splits_loaders_MNIST(train_size=None, val_size=10000, test_size=None, 
                              batch_size=100, test_batch_size=100,
                              device=None, force=False, **kwargs):
    """
    Loads the dynamically binarized MNIST trainset, applies
    a stratified split on it to form a train_loader/val_loader
    pair. Then loads the testset.
    Returns train_loader, val_loader, test loader
    """
    if "splitted_MNIST.pkl" not in os.listdir(str(PARENTFOLDER)) or force:
        print("\nMaking splits..")
        X, Y = load_binarized_mnist_tensor("train", batch_size=batch_size, device=device)
        if train_size is not None:
            skf = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=val_size)
        else:
            skf = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
        split_indices = skf.split(X, Y)

        train_index, val_index = [itm for itm in split_indices][0]

        train_X, train_Y = X[train_index,:], Y[train_index]
        val_X, val_Y = X[val_index,:], Y[val_index]

        train_dataset = TensorDataset(train_X, train_Y) 
        train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            **kwargs)
        
        val_dataset = TensorDataset(val_X, val_Y) 
        val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            **kwargs)

        if test_size is not None:
            X, Y = load_binarized_mnist_tensor("test", batch_size=test_batch_size, device=device, shuffle=False)

            skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            split_indices = skf.split(X, Y)

            test_index, _= [itm for itm in split_indices][0]

            test_X, test_Y = X[test_index,:], Y[test_index]

            test_dataset = TensorDataset(test_X, test_Y) 
            test_loader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                **kwargs)
        else:
            test_loader  = load_binarized_mnist_torch("test", 
                                                    batch_size=test_batch_size, 
                                                    device=device,
                                                    shuffle=False)

        # save for future use
        path = PARENTFOLDER / "splitted_MNIST.pkl"
        with open(str(path), "wb") as f:
            pkl.dump({"train": train_loader, 
                      "val": val_loader, 
                      "test": test_loader}, f)
    else:
        print("\nLoading pre-computed splits..")
        path = PARENTFOLDER / "splitted_MNIST.pkl"
        # load splits
        with open(str(path), "rb") as f:
            dico = pkl.load(f)
            train_loader, val_loader, test_loader = dico["train"], dico["val"], dico["test"]
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("Test loading MNIST")
    # RAPH: Authors dont do a CV they simply launch multiple runs on the same split
    # so a simple train/val/test is sufficient here and necessary to be closer to the paper
    # Note that the validation set is made of 10000 samples, just as in the tensorflow API
    # default batch_size is 100 as in the paper
    train_loader, val_loader, test_loader = make_splits_loaders_MNIST(force=True)
    
    print(len(train_loader), len(val_loader), len(test_loader))
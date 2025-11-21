# data_setup.py
import torch
import config
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_data(dataset_name='CIFAR10'):
    """Downloads and returns train/test datasets."""
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        num_classes = 10
        
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
        num_classes = 10
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    return train_ds, test_ds, num_classes


def get_non_iid_indices(train_ds, num_clients, alpha):
    """
    Splits training data into Non-IID shards using Dirichlet distribution.
    
    Args:
        train_ds: Training dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more skewed)
    
    Returns:
        List of index arrays, one per client
    """
    num_classes = len(train_ds.classes)
    labels = np.array(train_ds.targets)
    
    # Get indices for each class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    
    # Sample proportions from Dirichlet for each client
    client_class_proportions = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    # Distribute samples to clients
    client_indices_list = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        indices = class_indices[class_idx]
        np.random.shuffle(indices)
        
        # Split this class's data according to Dirichlet proportions
        proportions = client_class_proportions[:, class_idx]
        proportions = proportions / proportions.sum()  # Normalize
        
        splits = (proportions * len(indices)).astype(int)
        splits[-1] = len(indices) - splits[:-1].sum()  # Ensure all samples assigned
        
        start_idx = 0
        for client_idx, split_size in enumerate(splits):
            if split_size > 0:
                end_idx = start_idx + split_size
                client_indices_list[client_idx].extend(indices[start_idx:end_idx])
                start_idx = end_idx
    
    # Convert to numpy arrays and shuffle
    client_indices_list = [
        np.random.permutation(indices).tolist() 
        for indices in client_indices_list 
        if len(indices) > 0
    ]
    
    # Print distribution info
    print(f"Data distribution (Non-IID with α={alpha}):")
    for i, indices in enumerate(client_indices_list):
        client_labels = labels[indices]
        class_counts = [np.sum(client_labels == c) for c in range(num_classes)]
        print(f"  Client {i}: {len(indices)} samples, distribution: {class_counts}")
    
    return client_indices_list


def get_test_loader(test_ds):
    """Returns DataLoader for test set."""
    return DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
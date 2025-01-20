# utils.py
"""
Utility functions for data preparation, latent extraction, and visualization.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Data Loading

def load_dataset(batch_size=32, root="data"):
    """Load the EuroSAT dataset and create dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    # Load the dataset
    dataset = datasets.EuroSAT(root=root, download=True, transform=transform)

    # Split into train and test sets
    train_len = int(len(dataset) * 0.8)
    test_len = len(dataset) - train_len
    train_data, test_data = torch.utils.data.random_split(dataset, [train_len, test_len])

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Latent Extraction

def extract_latents(model, dataloader, device='cpu'):
    """Extract latent representations from a pre-trained VAE model."""
    model.eval()  # Set the model to evaluation mode
    latents = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch  # Assuming the first element of batch is input data
            inputs = inputs.to(device)
            outputs = model(inputs)
            latent = outputs["likelihoods"]["y"]

            # Flatten each latent in the batch before appending
            latents.append(latent.view(latent.size(0), -1).cpu().numpy())
            labels.append(label.cpu().numpy())

    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)

# Visualization

def plot_tsne_with_labels(latents, labels, perplexity=30, n_components=2, n_iter=500):
    """Visualize latent representations using t-SNE."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    T = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(T[:, 0], T[:, 1], c=labels, s=10, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Class Labels')
    plt.title('t-SNE visualization of VAE Latents with Labels')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

# Metric Calculation

def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM between original and reconstructed images."""
    psnr_value = psnr(original, reconstructed, data_range=1)
    ssim_value = ssim(original, reconstructed, channel_axis=2, data_range=1)
    return psnr_value, ssim_value

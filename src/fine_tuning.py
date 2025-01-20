# fine_tuning.py
"""
Fine-tuning workflow for Variational Autoencoders (VAEs) with composite loss functions.
"""
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import math

from models import get_pretrained_vae
from utils import load_dataset, calculate_metrics

# Fine-tuning function
def fine_tune_vae(model, train_loader, val_loader, epochs, lmbda, device):
    """Fine-tune a pre-trained VAE model with a composite loss function."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    aux_optimizer = optim.Adam(model.entropy_bottleneck.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (batch, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch = batch.to(device)

            # Zero gradients
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            x_hat = outputs["x_hat"]
            y_likelihoods = outputs["likelihoods"]["y"]

            # Calculate losses
            N, C, H, W = batch.size()
            num_pixels = N * H * W

            bpp_loss = torch.sum(torch.log(y_likelihoods)) / (-math.log(2) * num_pixels)
            mse_loss = F.mse_loss(batch, x_hat)
            loss = mse_loss + lmbda * bpp_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update entropy bottleneck
            aux_loss = model.entropy_bottleneck.loss()
            aux_loss.backward()
            aux_optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

        # Validate the model
        validate_vae(model, val_loader, device)

# Validation function
def validate_vae(model, val_loader, device):
    """Validate the fine-tuned VAE model."""
    model.eval()
    avg_psnr = 0.0
    avg_ssim = 0.0
    total_batches = len(val_loader)

    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            x_hat = outputs["x_hat"]

            # Compute metrics
            psnr_value, ssim_value = calculate_metrics(batch.cpu().numpy(), x_hat.cpu().numpy())
            avg_psnr += psnr_value
            avg_ssim += ssim_value

    avg_psnr /= total_batches
    avg_ssim /= total_batches
    print(f"Validation PSNR: {avg_psnr:.2f}, Validation SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    # Hyperparameters
    lmbda = 0.1
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, val_loader = load_dataset(batch_size=batch_size)

    # Load pre-trained VAE
    vae_model = get_pretrained_vae(model_name='bmshj2018_hyperprior', quality=5).to(device)

    # Fine-tune the VAE
    fine_tune_vae(vae_model, train_loader, val_loader, epochs, lmbda, device)

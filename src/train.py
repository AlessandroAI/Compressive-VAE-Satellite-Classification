# train.py
"""
Handles training workflow for classification models using latent representations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import get_pretrained_vae, TransformerClassifier
from utils import load_dataset, calculate_metrics, extract_latents

# Training function
def train_model(classifier, train_loader, val_loader, epochs, device):
    """Train a classification model on latent representations."""
    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, labels = data.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = classifier(data)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

        # Validate
        validate_model(classifier, val_loader, device)

# Validation function
def validate_model(classifier, val_loader, device):
    """Validate the classification model."""
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = classifier(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # Hyperparameters
    latent_dim = 192
    num_classes = 10
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, val_loader = load_dataset(batch_size=batch_size)

    # Load pre-trained VAE and extract latent representations
    vae_model = get_pretrained_vae(model_name='bmshj2018_hyperprior', quality=5).to(device)
    vae_model.eval()

    # Extract latents for training and validation
    train_latents, train_labels = extract_latents(vae_model, train_loader, device)
    val_latents, val_labels = extract_latents(vae_model, val_loader, device)

    # Prepare DataLoaders for latent space
    train_data = torch.utils.data.TensorDataset(torch.tensor(train_latents, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    val_data = torch.utils.data.TensorDataset(torch.tensor(val_latents, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Classification model
    classifier = TransformerClassifier(input_dim=latent_dim, num_classes=num_classes)

    # Train the classifier
    train_model(classifier, train_loader, val_loader, epochs, device)

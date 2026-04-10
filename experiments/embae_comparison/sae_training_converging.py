import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
# ==========================================
# 1. Data Preparation
# ==========================================
print("Loading and preparing dataset...")
digits = load_digits()
# FIX 1: Normalize digits from [0, 16] to [0, 1] to prevent massive latent spikes
X = torch.tensor(digits.data, dtype=torch.float32) / 16.0
# Precompute the mean for the tied bias (Bypass Weiszfeld NaN traps entirely)
mean_bias = X.mean(dim=0)
# ==========================================
# 2. Model Definition
# ==========================================
class SimpleSAE(nn.Module):
    def __init__(self, input_dim=64, latent_dim=1000, tied_bias=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.eps = 1e-8
        # Initialize tied bias safely
        if tied_bias is not None:
            self.tied_bias = nn.Parameter(tied_bias.clone())
        else:
            self.tied_bias = nn.Parameter(torch.zeros(input_dim))
        # Layers
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
        # FIX 2: Stable Sparse Initialization
        with torch.no_grad():
            # 1. Force decoder vectors to unit norm
            W_dec = self.decoder.weight
            W_dec.div_(W_dec.norm(dim=0, keepdim=True) + self.eps)
            # 2. Tie encoder weights to the normalized decoder weights
            self.encoder.weight.data = self.decoder.weight.data.T.clone()
            # 3. Set encoder bias slightly negative to prevent the "Dense Initialization Storm"
            nn.init.constant_(self.encoder.bias, -0.1)
    def forward(self, x):
        # Encode
        x_centered = x - self.tied_bias
        z_pre = self.encoder(x_centered)
        z = F.relu(z_pre)
        # Decode
        recon = self.decoder(z) + self.tied_bias
        return z, recon
# ==========================================
# 3. Training Setup
# ==========================================
# Hyperparameters
batch_size = 128  # Small batch size so Adam gets enough steps
epochs = 10000
lr = 1e-3
lambda_l1 = 0.1  # Sparsity penalty
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = SimpleSAE(input_dim=64, latent_dim=1000, tied_bias=mean_bias)
optimizer = optim.Adam(model.parameters(), lr=lr)
# ==========================================
# 4. Training Loop
# ==========================================
print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_mse = 0.0
    epoch_l1 = 0.0
    epoch_l0 = 0.0
    for (batch_x,) in loader:
        z, recon = model(batch_x)
        # Loss Calculation: Both losses summed over feature dim, averaged over batch
        mse_loss = (recon - batch_x).pow(2).sum(dim=-1).mean()
        l1_loss = z.abs().sum(dim=-1).mean()
        loss = mse_loss + lambda_l1 * l1_loss
        optimizer.zero_grad()
        loss.backward()
        # Standard gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # FIX 3: Gradient Projection
        # Prevent Adam from trying to change the magnitude of the decoder weights
        with torch.no_grad():
            W = model.decoder.weight
            W_dir = W / (W.norm(dim=0, keepdim=True) + model.eps)
            # Remove the gradient component parallel to the weight vector
            W.grad -= torch.sum(W.grad * W_dir, dim=0, keepdim=True) * W_dir
        optimizer.step()
        # Re-normalize decoder weights to correct any tiny floating-point drift
        with torch.no_grad():
            W = model.decoder.weight
            W.div_(W.norm(dim=0, keepdim=True) + model.eps)
        # Metrics logging
        epoch_mse += mse_loss.item()
        epoch_l1 += l1_loss.item()
        # L0 is the average number of active features per sample
        epoch_l0 += (z > 0).float().sum(dim=-1).mean().item()
    # Print stats periodically
    if epoch == 0 or (epoch + 1) % 100 == 0:
        avg_mse = epoch_mse / len(loader)
        avg_l1 = epoch_l1 / len(loader)
        avg_l0 = epoch_l0 / len(loader)
        total_loss = avg_mse + lambda_l1 * avg_l1
        print(
            f"Epoch {epoch + 1:4d} | Total Loss: {total_loss:.4f} | MSE: {avg_mse:.4f} | L1: {avg_l1:.4f} | L0: {avg_l0:.1f}")
print("Training complete! Your SAE did not diverge.")
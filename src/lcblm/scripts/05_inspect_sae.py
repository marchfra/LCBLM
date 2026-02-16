# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trained SAE inspection
#
# This notebook inspects the trained SAE.

# %% [markdown]
# ## 0 - Environment setup

# %% [markdown]
# #### Check that the notebook is running in Kaggle

# %%
from pathlib import Path

IN_KAGGLE = Path("/kaggle").exists()
if IN_KAGGLE:
    print("Running on Kaggle")
else:
    print("Not running on Kaggle")
    msg = "This notebook is intended to run on Kaggle only."
    raise SystemExit(msg)

# %% [markdown]
# #### Import libraries

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from better_kaggle_secrets import UserSecretsClient
from huggingface_hub import login as hf_login
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lcblm.sae_utils import SAEDataset, SparseAE, TopK
from lcblm.utils.plotting import set_plt_style
from lcblm.utils.seed import set_seeds

# %% [markdown]
# #### Set seed for reproducibility

# %%
SEED = 3742
set_seeds(SEED)

# %% [markdown]
# #### Setup matplotlib style

# %%
STYLE_PATH = Path("/kaggle/input/mpl-styles")
STYLES = ["grid", "science", "notebook", "mylegend"]

set_plt_style(styles=STYLES, style_path=STYLE_PATH)

# %% [markdown]
# #### Setup Telegram link

# %%
user_secrets = UserSecretsClient()
TG_TOKEN = user_secrets.get_secret("TELEGRAM_TOKEN")
TG_CHAT_ID = user_secrets.get_secret("TELEGRAM_CHAT_ID")

# %% [markdown]
# #### Setup Hugging Face link

# %%
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
if HF_TOKEN is not None:
    hf_login(token=HF_TOKEN)

# %% [markdown]
# #### Define output path

# %%
OUTPUT_PATH = Path("sae_inspection")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1 - Load data

# %%
EMBEDDINGS_PATH = Path("/kaggle/input/sst2-mistral-embeddings/sst2_mistral_embeddings")

if not EMBEDDINGS_PATH.exists():
    msg = f"Data path {EMBEDDINGS_PATH} does not exist."
    raise FileNotFoundError(msg)

val_data = torch.load(EMBEDDINGS_PATH / "extracted_features_validation.pt")

# %% [markdown]
# #### 1.1 - Load tokenizer

# %%
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
)

VOCAB_SIZE: int = tokenizer.vocab_size  # pyright: ignore[reportAssignmentType]
EOS_TOKEN_ID: int = tokenizer.eos_token_id  # pyright: ignore[reportAssignmentType]

# %% [markdown]
# #### 1.2 - Create dataset

# %%
val_dataset = SAEDataset(input_data=val_data["embeddings"])

# %% [markdown]
# ## 2 - Load trained model

# %%
TOP_K = 128  # TODO: read k from file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sae_state_dict = torch.load(
    "/kaggle/input/sae-on-mistral-embeddings/pytorch/default/1/best_sae_state.pt",
)
in_dimension = sae_state_dict["lin_encoder.weight"].shape[1]
latent_factor = sae_state_dict["lin_encoder.weight"].shape[0] // in_dimension
sae = SparseAE(
    input_dim=in_dimension,
    latent_dim=in_dimension * latent_factor,
    activation=nn.Sequential(TopK(k=TOP_K), nn.ReLU()),
)
sae.load_state_dict(sae_state_dict)
sae.to(device)
sae.eval()

# %% [markdown]
# ## 3 - Inspect SAE for dead latents
#
# A latent should be considered dead if and only if it's inactive for every sample *in the whole dataset* **and** for every token in the sample's context. This means that if a latent is dead, there is no word (or actually token) in any sentence (or actually context) that maps to that specific concept. Since this statement is confusing and quite possibly wrong, here's an example.
#
# - Dataset size: 2 (what a big dataset huh?)
# - Context window: 3
# - Number of latents: 5
#
# -> Latent shape: (2, 3, 5)
#
# ```
# latents = [[[0, 0, 0, 0, 5],
#             [0, 0, 3, 2, 1],
#             [0, 0, 3, 4, 5]],
#
#            [[0, 2, 0, 4, 5],
#             [0, 4, 3, 2, 1],
#             [0, 2, 3, 4, 5]]]
#
# is_dead =   [1, 0, 0, 0, 0]
# ```
#
# There are 5 neurons in the latent space:
# - latent 1 is dead (it's 0 for every token in the context of every sample)
# - latent 2 is alive (it's 0 for every token in sample 1, but not in sample 2)
# - latent 3 is alive (it's 0 for the first token in every sample, but not for all tokens)
# - latent 4 is alive (it's 0 for just one token in just one sample)
# - latent 5 is alive (it's never 0)

# %%
# Track activation counts for each latent
latent_dim = sae.latent_dim
context_window = val_dataset.input_data.shape[1]
activation_counts = torch.zeros(latent_dim, dtype=torch.long)
total_samples = len(val_dataset) * context_window

eps = 1e-12
# Pass validation data through the SAE
val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
)
recon_loss = 0.0
latents_l1_norm = 0.0
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Checking for dead latents", unit="batch"):
        embeddings = batch.to(device)
        output = sae(embeddings)
        latents = output.latents
        recon = output.recon

        recon_loss += F.mse_loss(recon, embeddings).item()
        latents_l1_norm += latents.abs().mean().item()

        # Count which latents were activated (non-zero)
        activated = (latents != 0).sum(
            dim=(0, 1),  # dim 0 = samples, dim 1 = tokens, dim 2 = concepts
        )
        activation_counts += activated.cpu().long()

    recon_loss /= len(val_loader)
    latents_l1_norm /= len(val_loader)

with (OUTPUT_PATH / "sae_stats.json").open("w") as f:
    stats = {
        "recon_loss": recon_loss,
        "latents_l1_norm": latents_l1_norm,
        "activation_counts": activation_counts.tolist(),
    }
    json.dump(stats, f, indent=4)

# Find dead latents (never activated)
dead_latents = (activation_counts == 0).nonzero(as_tuple=True)[0]

print(f"Total latent dimensions: {latent_dim}")
print(
    f"Dead latents (never activated): {len(dead_latents)} "
    f"({100 * len(dead_latents) / latent_dim:.2f}%)",
)

# %%
# Visualize activation distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of activation counts
ax1.hist(activation_counts.numpy(), bins=50, edgecolor="black", alpha=0.75)
ax1.set_xlabel("Number of Samples Activated")
ax1.set_ylabel("Number of Latents")
ax1.set_title("Distribution of Latent Activation Frequencies")
# ax1.set_xscale("log")
ax1.set_yscale("log")

# Sorted activation counts
sorted_counts = torch.sort(activation_counts, descending=True).values
ax2.plot(sorted_counts.numpy())
ax2.axvspan(
    xmin=len(activation_counts) - len(dead_latents),
    xmax=len(activation_counts),
    # color="black",  # colors[1],
    alpha=0.6,
    linewidth=0,
)
ax2.set_xlabel("Latent Index (sorted)")
ax2.set_ylabel("Number of Samples Activated")
# ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_title("Latent Activation Counts (Sorted)")

fig.tight_layout()
fig.savefig(OUTPUT_PATH / "latent_activations_on_val_set.png", dpi=300)
plt.show()

# %% [markdown]
# ## 4 - Compute correlation and coactivation matrices for the latents

# %%
import numpy as np

# Collect all latent activations
all_latents_ = []
with torch.inference_mode():
    for batch in tqdm(val_loader, desc="Collecting latents", unit="batch"):
        embeddings = batch.to(device)
        output = sae(embeddings)
        latents = output.latents
        # Flatten batch and sequence dims: (batch, seq, latent) -> (batch*seq, latent)
        all_latents_.append(latents.reshape(-1, latent_dim))

all_latents = torch.cat(all_latents_, dim=0)  # Shape: (N_samples, latent_dim)
del all_latents_

# %%
from typing import Literal

downsample_method: Literal["mean", "max"] = "max"
DOWNSAMPLE_SIZE = 500


# Strategy 1: Downsample the matrix for visualization
def downsample_matrix(
    matrix: np.ndarray,
    target_size: int = 1000,
    method: Literal["mean", "max"] = "mean",
) -> np.ndarray:
    """Downsample a large matrix by averaging/maxing blocks."""
    n = matrix.shape[0]
    if n <= target_size:
        return matrix

    block_size = n // target_size
    downsampled = np.zeros((target_size, target_size))

    for i in range(target_size):
        for j in range(target_size):
            i_start, i_end = i * block_size, (i + 1) * block_size
            j_start, j_end = j * block_size, (j + 1) * block_size
            if method == "mean":
                downsampled[i, j] = matrix[i_start:i_end, j_start:j_end].mean()
            elif method == "max":
                downsampled[i, j] = matrix[i_start:i_end, j_start:j_end].max()
            else:
                msg = f"Unknown downsampling method: {method}"
                raise ValueError(msg)

    return downsampled


# Strategy 2: Only plot summary statistics
def plot_matrix_summary(
    matrix: np.ndarray,
    title: str,
    output_path: str | Path,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Plot summary statistics instead of full matrix."""
    np.fill_diagonal(matrix, 0)

    # Compute statistics
    row_max = np.abs(matrix).max(axis=1)
    row_mean = np.abs(matrix).mean(axis=1)
    above_threshold = (np.abs(matrix) > threshold).sum(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Distribution of maximum values per latent
    axes[0, 0].hist(row_max, bins=50, edgecolor="black", alpha=0.75)
    axes[0, 0].set_xlabel("Max Absolute Value")
    axes[0, 0].set_ylabel("Number of Latents")
    axes[0, 0].set_title(f"Max {title} per Latent")
    axes[0, 0].set_yscale("log")

    # Distribution of mean values per latent
    axes[0, 1].hist(row_mean, bins=50, edgecolor="black", alpha=0.75)
    axes[0, 1].set_xlabel("Mean Absolute Value")
    axes[0, 1].set_ylabel("Number of Latents")
    axes[0, 1].set_title(f"Mean {title} per Latent")
    axes[0, 1].set_yscale("log")

    # Number of connections above threshold
    axes[1, 0].hist(above_threshold, bins=50, edgecolor="black", alpha=0.75)
    axes[1, 0].set_xlabel(f"# Connections > {threshold}")
    axes[1, 0].set_ylabel("Number of Latents")
    axes[1, 0].set_title(f"Latents with High {title}")
    axes[1, 0].set_yscale("log")

    # Sorted maximum values
    sorted_max = np.sort(row_max)[::-1]
    axes[1, 1].plot(sorted_max)
    axes[1, 1].set_xlabel("Latent Index (sorted)")
    axes[1, 1].set_ylabel(f"Max {title}")
    axes[1, 1].set_title(f"Sorted Max {title}")
    # axes[1, 1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.show()

    return {
        "mean_max": float(row_max.mean()),
        "median_max": float(np.median(row_max)),
        "mean_mean": float(row_mean.mean()),
        "latents_above_threshold": int((row_max > threshold).sum()),
    }


# %%
# Compute correlation matrix on GPU
# Standardize the data
mean = all_latents.mean(dim=0, keepdim=True)
std = all_latents.std(dim=0, keepdim=True)
standardized = (all_latents - mean) / (std + 1e-12)

# Compute correlation: corr(X, Y) = (X^T @ Y) / N
corr_matrix_gpu = (standardized.T @ standardized) / (all_latents.shape[0] - 1)

# Move to CPU only for visualization
corr_matrix = corr_matrix_gpu.cpu().numpy()

CORR_THRESHOLD = 0.5
np.fill_diagonal(corr_matrix, 0)
high_corr_pairs = np.argwhere(np.abs(corr_matrix) > CORR_THRESHOLD)

print(
    f"Found {len(high_corr_pairs)} latent pairs with |correlation| > {CORR_THRESHOLD}",
)

# For correlation matrix - use summary plots instead
corr_stats = plot_matrix_summary(
    corr_matrix,
    "Correlation",
    OUTPUT_PATH / "correlation_summary.png",
    threshold=CORR_THRESHOLD,
)

# Optionally: plot downsampled version for overview
downsampled_corr = downsample_matrix(
    np.abs(corr_matrix),
    target_size=DOWNSAMPLE_SIZE,
    method=downsample_method,
)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(downsampled_corr, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
ax.set_xlabel("Latent Index (downsampled)")
ax.set_ylabel("Latent Index (downsampled)")
ax.set_title(f"Absolute Correlation (Downsampled {DOWNSAMPLE_SIZE}x{DOWNSAMPLE_SIZE})")
plt.colorbar(im, ax=ax, label="|Correlation|")
fig.tight_layout()
fig.savefig(OUTPUT_PATH / "correlation_downsampled.png", dpi=300)
plt.show()

print(f"Correlation summary statistics: {corr_stats}")

del corr_matrix_gpu  # Free GPU memory

# %%
# Compute co-activation matrix on GPU
active_mask = (all_latents != 0).float()  # Shape: (N_samples, latent_dim)
coactivation_matrix_gpu = active_mask.T @ active_mask  # Shape: (latent_dim, latent_dim)

# Normalize by activation counts
activation_totals_gpu = activation_counts.to(device).float()[:, None]
normalized_coactivation_gpu = coactivation_matrix_gpu / (activation_totals_gpu + 1e-12)

# Move to CPU for visualization
normalized_coactivation = normalized_coactivation_gpu.cpu().numpy()

# Plot normalized coactivation matrix
COACT_THRESHOLD = 0.5
np.fill_diagonal(normalized_coactivation, 0)
high_coact_pairs = np.argwhere(normalized_coactivation > COACT_THRESHOLD)

print(
    f"Found {len(high_coact_pairs)} latent pairs with "
    f"co-activation > {COACT_THRESHOLD}",
)

# For co-activation matrix
coact_stats = plot_matrix_summary(
    normalized_coactivation,
    "Co-activation",
    OUTPUT_PATH / "coactivation_summary.png",
    threshold=COACT_THRESHOLD,
)

# Downsampled version
downsampled_coact = downsample_matrix(
    normalized_coactivation,
    target_size=DOWNSAMPLE_SIZE,
    method=downsample_method,
)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(downsampled_coact, cmap="viridis", aspect="auto", vmin=0, vmax=1)
ax.set_xlabel("Latent Index (downsampled)")
ax.set_ylabel("Latent Index (downsampled)")
ax.set_title(
    f"Co-activation Probability (Downsampled {DOWNSAMPLE_SIZE}x{DOWNSAMPLE_SIZE})",
)
plt.colorbar(im, ax=ax, label="Co-activation Probability")
fig.tight_layout()
fig.savefig(OUTPUT_PATH / "coactivation_downsampled.png", dpi=300)
plt.show()

print(f"Co-activation summary statistics: {coact_stats}")

del coactivation_matrix_gpu, normalized_coactivation_gpu  # Free GPU memory

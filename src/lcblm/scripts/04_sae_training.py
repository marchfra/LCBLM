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
# # SAE training on Mistral embeddings
#
# This notebook outputs the trained SAE along with the learning curves and losses and a latent activation visualisation.

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
from sae_utils import Config, SAEDataset, train_sae
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lcblm.utils.plotting import plot_learning_curves, set_plt_style
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
# #### Define output path

# %%
OUTPUT_PATH = Path("sae")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1 - Load data

# %%
EMBEDDINGS_PATH = Path("/kaggle/input/sst2-mistral-embeddings")
SPLITS = ["train", "validation"]

if not EMBEDDINGS_PATH.exists():
    msg = f"Data path {EMBEDDINGS_PATH} does not exist."
    raise FileNotFoundError(msg)

data = {
    split: torch.load(EMBEDDINGS_PATH / f"extracted_features_{split}.pt")
    for split in SPLITS
}

# %% [markdown]
# #### 1.1 - Load tokenizer

# %%
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
)

VOCAB_SIZE: int = tokenizer.vocab_size  # pyright: ignore[reportAssignmentType] # Mistral's vocabulary size
EOS_TOKEN_ID: int = tokenizer.eos_token_id  # pyright: ignore[reportAssignmentType] # End-of-sequence token ID for Mistral

# %% [markdown]
# #### 1.2 - Create dataset

# %%
datasets = {split: SAEDataset(data=data[split]["embeddings"]) for split in SPLITS}

# %% [markdown]
# ## 3 - Train SAE

# %%
config = Config(
    n_epochs=200,  # Training time is about 1m 30s per epoch
    batch_size=256,
    learning_rate=1e-4,
    latent_dim_factor=4,
    k=128,
    threshold_dead_latent=1_500,  # with 200 epochs and 28 batches there is a total of
    # 200*28 = 5600 batches, so a threshold of 25_000 would be useless.
    k_aux=512,  # consider trying 256 as well
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
sae, epoch_losses, batch_losses, val_losses, best_epoch = train_sae(
    config,
    datasets["train"],
    datasets["validation"],
)
print("Saving trained SAE...")
torch.save(
    sae.module.state_dict(),
    OUTPUT_PATH / f"sae_on_mistral_sst2-{config.n_epochs}epochs.pt",
)
torch.save(
    {
        "epoch_losses": epoch_losses,
        "val_losses": val_losses,
        "batch_losses": batch_losses,
    },
    OUTPUT_PATH / "training_losses.pt",
)
print("✅ Trained SAE and training losses saved!")

# %% [markdown]
# ### 3.1 - Plotting

# %%
plot_learning_curves(
    epoch_losses,
    val_losses,
    best_epoch,
    output_path=OUTPUT_PATH,
    tg_token=TG_TOKEN,
    tg_chat_id=TG_CHAT_ID,
    caption="SAE final curves",
)

fig_batch, ax = plt.subplots(1, 1)
ax.plot(range(1, len(batch_losses) + 1), batch_losses, label="Batch Loss")

ax.set_xlabel("Batch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Over Batches")
ax.legend()

fig_batch.tight_layout()
fig_batch.savefig(OUTPUT_PATH / "training_loss_over_batches.png", dpi=300)
plt.show()

# %% [markdown]
# ## 4 - Load trained model

# %%
sae.module.load_state_dict(
    torch.load(OUTPUT_PATH / f"sae_on_mistral_sst2-{config.n_epochs}epochs.pt"),
)
sae.eval()

# %% [markdown]
# ### 4.1 - Inspect trained model for dead latents
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
latent_dim = sae.module.latent_dim
context_window = datasets["validation"].data.shape[1]
activation_counts = torch.zeros(latent_dim, dtype=torch.long)
total_samples = len(datasets["validation"]) * context_window

eps = 1e-12
# Pass validation data through the SAE
val_loader = DataLoader(
    datasets["validation"],
    batch_size=config.batch_size,
    shuffle=False,
)
recon_loss = 0.0
latents_l1_norm = 0.0
latents_entropy = 0.0
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Checking for dead latents", unit="batch"):
        embeddings = batch.to(device)
        output = sae(embeddings)
        latents = output.latents
        recon = output.recon

        recon_loss += F.mse_loss(recon, embeddings).item()
        latents_l1_norm += latents.abs().mean().item()

        # Check TopK activation
        assert ((latents != 0).sum(dim=2) == 128).all()  # noqa: PLR2004, S101

        # Count which latents were activated (non-zero)
        activated = (latents != 0).sum(
            dim=(0, 1),  # dim 0 = samples, dim 1 = tokens, dim 2 = concepts
        )
        activation_counts += activated.cpu().long()

with (OUTPUT_PATH / "sae_stats.json").open("w") as f:
    stats = {
        "recon_loss": recon_loss / len(val_loader),
        "latents_l1_norm": latents_l1_norm / len(val_loader),
        "activation_counts": activation_counts.tolist(),
    }
    json.dump(stats, f, indent=4)

# Find dead latents (never activated)
dead_latents = (activation_counts == 0).nonzero(as_tuple=True)[0]

print(f"\nTotal latent dimensions: {latent_dim}")
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
ax2.set_xlabel("Latent Index (sorted)")
ax2.set_ylabel("Number of Samples Activated")
ax2.set_xscale("log")
# ax2.set_yscale("log")
ax2.set_title("Latent Activation Counts (Sorted)")

fig.tight_layout()
fig.savefig(OUTPUT_PATH / "latent_activations_on_val_set.png", dpi=300)
plt.show()

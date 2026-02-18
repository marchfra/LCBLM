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
# # Bernoulli SAE training on Mistral embeddings
#
# This notebook outputs the trained SAE along with the learning curves and losses.

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
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from better_kaggle_secrets import UserSecretsClient
from huggingface_hub import login as hf_login
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from trainvox import (
    CompositeStrategy,
    PrintStrategy,
    TelegramTqdmStrategy,
    TqdmStrategy,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lcblm.sae_utils import SAEDataset, SparseAE
from lcblm.sae_utils.dataset import compute_tied_bias
from lcblm.sae_utils.losses import bernoulli_kl_loss
from lcblm.utils.data import typed_dataloader
from lcblm.utils.plotting import (
    learning_curves_plot,
    send_learning_curves_to_telegram,
    set_plt_style,
)
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
OUTPUT_PATH = Path("bernoulli_sae")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1 - Load data

# %%
EMBEDDINGS_PATH = Path("/kaggle/input/sst2-mistral-embeddings/sst2_mistral_embeddings")
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

VOCAB_SIZE: int = tokenizer.vocab_size  # pyright: ignore[reportAssignmentType]
EOS_TOKEN_ID: int = tokenizer.eos_token_id  # pyright: ignore[reportAssignmentType]

# %% [markdown]
# #### 1.2 - Create dataset

# %%
datasets = {
    split: SAEDataset(input_data=data[split]["embeddings"].to(dtype=torch.float32))
    for split in SPLITS
}

# %% [markdown]
# ## 2 - Define SAE and training parameters

# %%
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100  # about 1m 9s per epoch
BATCH_SIZE = 256
LATENT_DIM_FACTOR = 4
LAMBDA_KL = 0.3
LAMBDA_L1 = 0
BERNOULLI_P = 0.05

y_label = "MSE"
if LAMBDA_KL != 0:
    y_label += f" + {LAMBDA_KL} * KL"
if LAMBDA_L1 != 0:
    y_label += f" + {LAMBDA_L1} * L1"
y_label += " Loss"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

sae = SparseAE(
    input_dim=datasets["train"].num_features,
    latent_dim=datasets["train"].num_features * LATENT_DIM_FACTOR,
    activation=nn.LeakyReLU(),
)
sae.init_tied_bias(compute_tied_bias(datasets["train"]))
sae.to(device)

recon_criterion = nn.MSELoss()
kl_criterion = partial(bernoulli_kl_loss, target_p=BERNOULLI_P)
optimizer = Adam(sae.parameters(), lr=LEARNING_RATE)

# %%
dataloaders = {
    split: DataLoader(
        datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
    )
    for split in SPLITS
}

# %% [markdown]
# ## 3 - Train SAE

# %%
LEARNING_CURVES_PATH = OUTPUT_PATH / "learning_curves.png"

if TG_TOKEN and TG_CHAT_ID:
    v = CompositeStrategy(
        TelegramTqdmStrategy(token=TG_TOKEN, chat_id=TG_CHAT_ID),
        PrintStrategy(),
    )
else:
    v = CompositeStrategy(
        TqdmStrategy(),
        PrintStrategy(),
    )

v.on_train_begin(
    NUM_EPOCHS,
    msg="Starting training of *Bernoulli SAE*",
)
training_losses: list[float] = []
validation_losses: list[float] = []
batch_losses: list[float] = []

train_recon_losses: list[float] = []
train_kl_losses: list[float] = []
train_l1_losses: list[float] = []
val_recon_losses: list[float] = []
val_kl_losses: list[float] = []
val_l1_losses: list[float] = []

best_val_loss = float("inf")
best_sae_state = sae.state_dict()
best_epoch = -1
msg_id: int | None = None
for epoch in v.wrap_epoch_iterator(range(NUM_EPOCHS)):
    epoch_loss = 0.0
    recon_loss_ = 0.0
    kl_loss_ = 0.0
    l1_loss_ = 0.0
    sae.train()
    for batch in typed_dataloader(dataloaders["train"]):
        x = batch.reshape(-1, sae.input_dim).to(device, dtype=torch.float32)
        latents, latents_pre_act, recon, _ = sae(x)

        recon_loss = recon_criterion(x, recon)
        kl_loss = kl_criterion(latents_pre_act)
        l1_loss = torch.abs(latents).mean()
        loss = recon_loss + LAMBDA_KL * kl_loss + LAMBDA_L1 * l1_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.item())
        epoch_loss += loss.item()
        recon_loss_ += recon_loss.item()
        kl_loss_ += kl_loss.item()
        l1_loss_ += l1_loss.item()

    epoch_loss /= len(dataloaders["train"])
    training_losses.append(epoch_loss)
    train_recon_losses.append(recon_loss_ / len(dataloaders["train"]))
    train_kl_losses.append(kl_loss_ / len(dataloaders["train"]))
    train_l1_losses.append(l1_loss_ / len(dataloaders["train"]))

    val_loss = 0.0
    recon_loss_ = 0.0
    kl_loss_ = 0.0
    l1_loss_ = 0.0
    sae.eval()
    with torch.inference_mode():
        for batch in typed_dataloader(dataloaders["validation"]):
            x = batch.reshape(-1, sae.input_dim).to(device, dtype=torch.float32)
            latents, latents_pre_act, recon, _ = sae(x)

            recon_loss = recon_criterion(x, recon)
            kl_loss = kl_criterion(latents_pre_act)
            l1_loss = torch.abs(latents).mean()
            loss = recon_loss + LAMBDA_KL * kl_loss + LAMBDA_L1 * l1_loss

            val_loss += loss.item()
            recon_loss_ += recon_loss.item()
            kl_loss_ += kl_loss.item()
            l1_loss_ += l1_loss.item()

    val_loss /= len(dataloaders["validation"])
    validation_losses.append(val_loss)
    val_recon_losses.append(recon_loss_ / len(dataloaders["validation"]))
    val_kl_losses.append(kl_loss_ / len(dataloaders["validation"]))
    val_l1_losses.append(l1_loss_ / len(dataloaders["validation"]))

    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = val_loss
        best_sae_state = sae.state_dict()
        torch.save(best_sae_state, OUTPUT_PATH / "best_sae_state.pt")

    if epoch % 2 == 0 and epoch != 0:
        with learning_curves_plot(
            training_losses,
            validation_losses,
            title="Bernoulli SAE Learning Curves",
            best_epoch=best_epoch,
        ) as (fig, ax):
            ax.set_ylabel(y_label)
            fig.savefig(LEARNING_CURVES_PATH, dpi=300)
            if TG_TOKEN is not None and TG_CHAT_ID is not None:
                msg_id = send_learning_curves_to_telegram(
                    image_path=LEARNING_CURVES_PATH,
                    tg_token=TG_TOKEN,
                    tg_chat_id=TG_CHAT_ID,
                    caption="Bernoulli SAE intermediate curves",
                    msg_id=msg_id,
                )
            else:
                plt.show()
    v.on_epoch_end(epoch, train_loss=epoch_loss, val_loss=val_loss)

v.on_train_end()

with (OUTPUT_PATH / "losses.json").open("w") as f:
    json.dump(
        {
            "training_losses": training_losses,
            "training_recon": train_recon_losses,
            "training_kl": train_kl_losses,
            "training_l1": train_l1_losses,
            "validation_losses": validation_losses,
            "validation_recon": val_recon_losses,
            "validation_kl": val_kl_losses,
            "validation_l1": val_l1_losses,
            "batch_losses": batch_losses,
        },
        f,
    )

# %%
with learning_curves_plot(
    training_losses,
    validation_losses,
    title="Bernoulli SAE Learning Curves",
    best_epoch=best_epoch,
) as (fig, ax):
    ax.set_ylabel(y_label)
    # ax.set_yscale("log")
    fig.savefig(LEARNING_CURVES_PATH, dpi=300)
    if TG_TOKEN is not None and TG_CHAT_ID is not None:
        msg_id = send_learning_curves_to_telegram(
            image_path=LEARNING_CURVES_PATH,
            tg_token=TG_TOKEN,
            tg_chat_id=TG_CHAT_ID,
            caption="Bernoulli SAE final curves",
            msg_id=msg_id,
        )
    plt.show()

with learning_curves_plot(
    batch_losses,
    title="Bernoulli SAE Training Loss Over Batches",
) as (fig, ax):
    ax.set_xlabel("Batch")
    ax.set_ylabel(y_label)
    # ax.set_yscale("log")
    fig.savefig(OUTPUT_PATH / "training_loss_over_batches.png", dpi=300)
    if TG_TOKEN is not None and TG_CHAT_ID is not None:
        send_learning_curves_to_telegram(
            OUTPUT_PATH / "training_loss_over_batches.png",
            tg_token=TG_TOKEN,
            tg_chat_id=TG_CHAT_ID,
            caption="Bernoulli SAE training loss over batches",
        )
    plt.show()

# %%
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


def _configure_integer_ticks(ax: Axes) -> None:
    """Set up integer major ticks with appropriately spaced minor ticks."""
    ax.xaxis.set_major_locator(MaxNLocator("auto", integer=True))
    major_ticks = ax.xaxis.get_majorticklocs()
    if len(major_ticks) >= 2:  # noqa: PLR2004
        # Compute minor ticks after major ticks are placed
        major_spacing = int(major_ticks[1] - major_ticks[0])
        # Find a nice divisor of major_spacing for minor ticks
        divisors = [
            d
            for d in [5, 4, 3, 2]
            if major_spacing % d == 0 and major_spacing // d >= 1
        ]
        n_minor = divisors[0] if divisors else major_spacing

        ax.xaxis.set_minor_locator(AutoMinorLocator(n_minor))


# %%
fig, ax = plt.subplots()

ax.plot(range(1, len(train_recon_losses) + 1), train_recon_losses, label="Train MSE")
ax.plot(range(1, len(train_kl_losses) + 1), train_kl_losses, label="Train KL")
ax.plot(range(1, len(train_l1_losses) + 1), train_l1_losses, label="Train L1")
ax.plot(range(1, len(val_recon_losses) + 1), val_recon_losses, label="Val MSE", ls="--")
ax.plot(range(1, len(val_kl_losses) + 1), val_kl_losses, label="Val KL", ls="--")
ax.plot(range(1, len(val_l1_losses) + 1), val_l1_losses, label="Val L1", ls="--")

_configure_integer_ticks(ax)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Bernoulli SAE Losses Breakdown")
ax.legend(
    ncol=2,
    title=r"$\lambda_{KL} = $"
    rf"${LAMBDA_KL}$, "
    r"$\lambda_{L_1} = $"
    rf"${LAMBDA_L1}$",
)
# ax.set_yscale("log")

fig.tight_layout()
fig.savefig(OUTPUT_PATH / "losses_breakdown.png", dpi=300)
if TG_TOKEN is not None and TG_CHAT_ID is not None:
    send_learning_curves_to_telegram(
        OUTPUT_PATH / "losses_breakdown.png",
        tg_token=TG_TOKEN,
        tg_chat_id=TG_CHAT_ID,
        caption="Bernoulli SAE Losses Breakdown",
    )
plt.show()
plt.close(fig)

# %%
sae.eval()
with torch.inference_mode():
    for batch in typed_dataloader(dataloaders["validation"]):
        x = batch.reshape(-1, sae.input_dim).to(device, dtype=torch.float32)
        _, latents_pre_act, *_ = sae(x)

        average_concept_activations = (
            torch.sigmoid(latents_pre_act).mean(dim=0).cpu().numpy()
        )

# %%
import numpy as np

# 1. Histogram with BERNOULLI_P line
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(average_concept_activations, bins=50, edgecolor="black", alpha=0.75)
axes[0].axvline(
    BERNOULLI_P,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Target: {BERNOULLI_P}",
)
axes[0].set_xlabel("Probability")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Bernoulli Probabilities")
axes[0].legend()
axes[0].set_yscale("log")


# 2. Summary statistics box
deviations = average_concept_activations - BERNOULLI_P
stats_text = f"""Summary Statistics:
Target: {BERNOULLI_P:.4f}
Mean: {average_concept_activations.mean():.4f}
Median: {np.median(average_concept_activations):.4f}
Std: {average_concept_activations.std():.4f}

Within ±10%: {np.sum(np.abs(deviations) < 0.1 * BERNOULLI_P) / len(average_concept_activations) * 100:.1f}%
Within ±20%: {np.sum(np.abs(deviations) < 0.2 * BERNOULLI_P) / len(average_concept_activations) * 100:.1f}%

Min: {average_concept_activations.min():.4f}
Max: {average_concept_activations.max():.4f}"""

axes[1].text(
    0.35,
    0.5,
    stats_text,
    fontsize=10,
    verticalalignment="center",
    family="monospace",
    bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
)
axes[1].axis("off")

fig.tight_layout()
fig.savefig(OUTPUT_PATH / "sae_inspection.png", dpi=300)
if TG_TOKEN is not None and TG_CHAT_ID is not None:
    send_learning_curves_to_telegram(
        OUTPUT_PATH / "sae_inspection.png",
        tg_token=TG_TOKEN,
        tg_chat_id=TG_CHAT_ID,
        caption="Bernoulli SAE Inspection",
    )
plt.show()
plt.close(fig)

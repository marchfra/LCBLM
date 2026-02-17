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
# # TopK SAE training on Mistral embeddings
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
from pathlib import Path

import torch
from better_kaggle_secrets import UserSecretsClient
from huggingface_hub import login as hf_login
from trainvox import (
    CompositeStrategy,
    PrintStrategy,
    TelegramTqdmStrategy,
    TqdmStrategy,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lcblm.sae_utils import Config, SAEDataset, train_sae
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
OUTPUT_PATH = Path("topk_sae")
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
# ## 3 - Train SAE

# %%
config = Config(
    n_epochs=200,  # Training time is about 1m 34s per epoch
    batch_size=256,
    learning_rate=1e-4,
    latent_dim_factor=4,
    k=128,
    # With 200 epochs and 28 batches there is a total of 200*28 = 5600 batches, so we
    # need a threshold smaller than that
    threshold_dead_latent=1000,
    k_aux=512,  # consider trying 256 as well
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
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

sae, epoch_losses, batch_losses, val_losses, best_epoch = train_sae(
    config,
    datasets["train"],
    datasets["validation"],
    verbosity_strategy=v,
)
print("Saving trained SAE...")
torch.save(sae.state_dict(), OUTPUT_PATH / "best_sae_state.pt")
with (OUTPUT_PATH / "losses.json").open("w") as f:
    json.dump(
        {
            "training_losses": epoch_losses,
            "validation_losses": val_losses,
            "batch_losses": batch_losses,
        },
        f,
    )
print("✅ Trained SAE and training losses saved!")

# %% [markdown]
# ### 3.1 - Plotting

# %%
LEARNING_CURVES_PATH = OUTPUT_PATH / "learning_curves.png"
with learning_curves_plot(
    epoch_losses,
    val_losses,
    title="TopK SAE Learning Curves",
    best_epoch=best_epoch,
) as (fig, _):
    fig.savefig(LEARNING_CURVES_PATH, dpi=300)
    if TG_TOKEN is not None and TG_CHAT_ID is not None:
        send_learning_curves_to_telegram(
            LEARNING_CURVES_PATH,
            tg_token=TG_TOKEN,
            tg_chat_id=TG_CHAT_ID,
            caption="Final TopK SAE learning curves",
        )

with learning_curves_plot(
    batch_losses,
    title="TopK SAE Training Loss Over Batches",
) as (fig, ax):
    ax.set_xlabel("Batch")
    fig.savefig(OUTPUT_PATH / "training_loss_over_batches.png", dpi=300)
    if TG_TOKEN is not None and TG_CHAT_ID is not None:
        send_learning_curves_to_telegram(
            OUTPUT_PATH / "training_loss_over_batches.png",
            tg_token=TG_TOKEN,
            tg_chat_id=TG_CHAT_ID,
            caption="TopK SAE training loss over batches",
        )

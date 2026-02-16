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
# # Retrain Mistral classifier head on SST2 embeddings
#
# This notebook trains a linear classifier head from scratch on top of precomputed Mistral embeddings for the SST2 dataset.
#
# The output of this notebook is the state dict of the trained classifier head with the lowest validation loss, saved as `retrained_classifier_head/best_classifier_state.pt`, along with the training and validation losses saved as `retrained_classifier_head/losses.json` and the learning curves plot `retrained_classifier_head/losses.png`.

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
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from better_kaggle_secrets import UserSecretsClient
from huggingface_hub import login as hf_login
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from trainvox import (
    CompositeStrategy,
    PrintStrategy,
    TelegramTqdmStrategy,
    TqdmStrategy,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lcblm.utils.data import NextTokenDataset, Sentence, typed_dataloader
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
OUTPUT_PATH = Path("retrained_classifier_head")
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
datasets: dict[str, NextTokenDataset] = {
    split: NextTokenDataset(
        input_ids=data[split]["input_ids"],
        attention_mask=data[split]["attention_masks"],
        embeddings=data[split]["embeddings"].float(),
        eos_token_id=EOS_TOKEN_ID,
    )
    for split in SPLITS
}

# %% [markdown]
# ## 2 - Define classifier and training parameters

# %%
LEARNING_RATE = 5e-4
NUM_EPOCHS = 20
WARMUP_EPOCHS = 10
BATCH_SIZE = 256

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(datasets[SPLITS[0]].embedding_dimension, VOCAB_SIZE, bias=False),
).to(device)
print(classifier)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=LEARNING_RATE)


def lr_lambda_warmup(current_epoch: int) -> float:
    """Learning rate schedule with linear warmup and cosine decay."""
    cosine_epochs = NUM_EPOCHS - WARMUP_EPOCHS

    # Linear warmup
    if current_epoch < WARMUP_EPOCHS:
        return (current_epoch + 1) / max(1, WARMUP_EPOCHS)

    # After warmup do cosine decay
    progress = float(current_epoch - WARMUP_EPOCHS) / float(max(1, cosine_epochs))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1)))
    return cosine_decay  # This factor is multiplied by the initial_lr


scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

# %%
dataloaders: dict[str, DataLoader[Sentence]] = {
    split: DataLoader(
        datasets[split],
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
    )
    for split in SPLITS
}

# %% [markdown]
# ## 3 - Train classifier

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
    msg="Starting training of *classifier head from scratch*",
)
training_losses: list[float] = []
validation_losses: list[float] = []
best_val_loss = float("inf")
best_classifier_state = classifier.state_dict()
best_epoch = -1
msg_id: int | None = None
for epoch in v.wrap_epoch_iterator(range(NUM_EPOCHS)):
    epoch_loss = 0.0
    classifier.train()
    for batch in typed_dataloader(dataloaders["train"]):
        embeddings_batch = batch.embeddings.to(device)
        next_token_ids_batch = batch.next_token_ids.to(device)
        next_attention_mask_batch = batch.next_attention_mask.to(
            device,
            dtype=torch.bool,
        )

        logits = classifier(embeddings_batch)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1])[next_attention_mask_batch.reshape(-1)],
            next_token_ids_batch.reshape(-1)[next_attention_mask_batch.reshape(-1)],
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloaders["train"])
    training_losses.append(epoch_loss)

    val_loss = 0.0
    classifier.eval()
    with torch.inference_mode():
        for batch in typed_dataloader(dataloaders["validation"]):
            embeddings_batch = batch.embeddings.to(device)
            next_token_ids_batch = batch.next_token_ids.to(device)
            next_attention_mask_batch = batch.next_attention_mask.to(
                device,
                dtype=torch.bool,
            )

            logits = classifier(embeddings_batch)
            loss = criterion(
                logits.reshape(-1, logits.shape[-1])[
                    next_attention_mask_batch.reshape(-1)
                ],
                next_token_ids_batch.reshape(-1)[next_attention_mask_batch.reshape(-1)],
            )

            val_loss += loss.item()

    val_loss /= len(dataloaders["validation"])
    validation_losses.append(val_loss)

    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = val_loss
        best_classifier_state = classifier.state_dict()
        torch.save(best_classifier_state, OUTPUT_PATH / "best_classifier_state.pt")

    scheduler.step()
    if epoch % 2 == 0 and epoch != 0:
        with learning_curves_plot(
            training_losses,
            validation_losses,
            title="Retrained Classifier Head Learning Curves",
            best_epoch=best_epoch,
        ) as (fig, ax):
            ax.set_ylabel("CE Loss")
            fig.savefig(LEARNING_CURVES_PATH, dpi=300)
            if TG_TOKEN is not None and TG_CHAT_ID is not None:
                msg_id = send_learning_curves_to_telegram(
                    image_path=LEARNING_CURVES_PATH,
                    tg_token=TG_TOKEN,
                    tg_chat_id=TG_CHAT_ID,
                    caption="Retrained head intermediate curves",
                    msg_id=msg_id,
                )
    v.on_epoch_end(epoch, train_loss=epoch_loss, val_loss=val_loss)

v.on_train_end()

with (OUTPUT_PATH / "losses.json").open("w") as f:
    json.dump(
        {"training_losses": training_losses, "validation_losses": validation_losses},
        f,
    )

# %%
with learning_curves_plot(
    training_losses,
    validation_losses,
    title="Retrained Classifier Head Learning Curves",
    best_epoch=best_epoch,
) as (fig, ax):
    ax.set_ylabel("CE Loss")
    fig.savefig(LEARNING_CURVES_PATH, dpi=300)
    if TG_TOKEN is not None and TG_CHAT_ID is not None:
        msg_id = send_learning_curves_to_telegram(
            image_path=LEARNING_CURVES_PATH,
            tg_token=TG_TOKEN,
            tg_chat_id=TG_CHAT_ID,
            caption="Retrained head final curves",
            msg_id=msg_id,
        )
    plt.show()

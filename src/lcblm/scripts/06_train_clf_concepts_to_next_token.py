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

# %% [markdown] _cell_guid="b94ca716-6dcd-4e74-82d9-1168b6e37bda" _uuid="41c544ab-b8cb-4611-ac78-9aceb92ce5c5" jupyter={"outputs_hidden": false}
# # Classifier training on concept representations for next-token prediction
#
# Starting from sst2 I need to:
# 1. tokenize the sentences with Mistral's tokenizer
# 2. pass the tokenized sentences through Mistral to get the embeddings at the last layer
#     - in the future, get the embeddings from various layers
#     - train a classifier to predict the token ids from the embeddings as a sanity check
# 3. train a SAE to map from the embedding of the last layer to concept space
#     - in the future, train a SAE to map from various layers to concept space
# 4. extract the concept representations for each token in the sentence
# 5. train a linear classifier to predict the *next* token id from the concept representation of the *current* token
#
# The focus of this notebook is step 5.
#
# <!-- This notebook outputs the trained SAE along with the learning curves and losses and a latent activation visualisation. -->

# %% [markdown] _cell_guid="67287a37-b4ab-4545-a565-d1f0b6d45e9c" _uuid="8e5f3f45-5f77-4a40-9b70-0c4c0714e3b9" jupyter={"outputs_hidden": false}
# ## 0 - Environment Setup

# %% [markdown] _cell_guid="ea0b196f-01b1-4ed8-b67a-5be79749646a" _uuid="85b6060d-71ad-48b1-9102-ceff09324529" jupyter={"outputs_hidden": false}
# #### Check that the notebook is running in Kaggle

# %% _cell_guid="bf095c53-8de4-4f8e-852d-ff606633b263" _uuid="4fd38baf-0ad1-4090-9911-5099ed21dfb1" jupyter={"outputs_hidden": false}
from pathlib import Path

IN_KAGGLE = Path("/kaggle").exists()
if IN_KAGGLE:
    print("Running on Kaggle")
else:
    print("Not running on Kaggle")
    msg = "This notebook is intended to run on Kaggle only."
    raise SystemExit(msg)

# %% [markdown] _cell_guid="765703ec-10c0-457c-bb81-10cc01771700" _uuid="5494c73b-f21c-42d0-a711-74c36672956d" jupyter={"outputs_hidden": false}
# ### 0.1 - Import libraries

# %% _cell_guid="eda13012-8054-4391-8b14-c0e5dc6a8076" _uuid="7dc3be8b-265a-4ae0-a41e-5d059fd3153f" jupyter={"outputs_hidden": false}
import math
import random
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import torch
from better_kaggle_secrets import UserSecretsClient
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from trainvox import (
    CompositeStrategy,
    PrintStrategy,
    TelegramTqdmStrategy,
    TqdmStrategy,
    send_telegram_photo,
)
from transformers import AutoTokenizer

torch.manual_seed(3742)
random.seed(3742)

# %% _cell_guid="64985136-544d-4d53-af6e-9864128ccc9f" _uuid="59ab1598-ba16-4fbd-8df2-3084f8b2644e" jupyter={"outputs_hidden": false}
STYLES_PATH = Path("/kaggle/input/mpl-styles")
STYLES = ["grid", "science", "notebook", "mylegend"]

if STYLES_PATH.exists():
    try:
        plt.style.use([str(STYLES_PATH / f"{style}.mplstyle") for style in STYLES])
    except FileNotFoundError:
        print("Some style files not found, using default matplotlib style.")
else:
    print("Styles path not found, using default matplotlib style.")
    plt.style.use("default")

# %% [markdown] _cell_guid="0c74aef7-52af-420a-9b43-f1dddf123b9a" _uuid="10c3d1ab-2c21-4691-b59e-68e41614dd0d" jupyter={"outputs_hidden": false}
# #### Setup Telegram link

# %% _cell_guid="5d56351d-8fe2-4eca-8644-5f9fff5afa3e" _uuid="ef487cc3-bcc4-4268-840e-c5195b88ac18" jupyter={"outputs_hidden": false}
user_secrets = UserSecretsClient()
TG_TOKEN = user_secrets.get_secret("TELEGRAM_TOKEN")
TG_CHAT_ID = user_secrets.get_secret("TELEGRAM_CHAT_ID")

# %% [markdown] _cell_guid="696e7b27-fbf2-498f-aabc-c4546a192f3d" _uuid="917f85df-7fab-4f3f-9a1e-a7eae45c23ad" jupyter={"outputs_hidden": false}
# ## 1 - Load Data

# %% _cell_guid="3a2f8a0d-2755-4965-b66a-bca0dec41b2b" _uuid="d0d26f22-93a2-472c-92a7-da8dc71f45eb" jupyter={"outputs_hidden": false}
LATENTS_PATH = Path("/kaggle/input/sst2-mistral-concepts-dense")

if not LATENTS_PATH.exists():
    msg = f"Data path {LATENTS_PATH} does not exist."
    raise FileNotFoundError(msg)

train_chunks_paths = list(LATENTS_PATH.glob("train_latents_chunk_*.pt"))
val_chunks_paths = list(LATENTS_PATH.glob("val_latents_chunk_*.pt"))
test_chunks_paths = list(LATENTS_PATH.glob("test_latents_chunk_*.pt"))

# %% [markdown]
# #### 1.1 - Load tokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7b-v0.1")

# %% [markdown] _cell_guid="c840df17-47c6-43bf-b4fe-ca55823438e6" _uuid="a70dfc68-aca2-4219-8593-e9e2fc6e9130" jupyter={"outputs_hidden": false}
# #### 1.1 - Define dataset

# %% _cell_guid="87e68440-d863-486e-8b4f-4723c7ed08b7" _uuid="f4d03294-f3dd-443b-ad84-7195725590b8" jupyter={"outputs_hidden": false}
NUM_CONCEPTS = 16384  # Number of concepts extracted by SAE
VOCAB_SIZE: int = tokenizer.vocab_size  # Mistral's vocabulary size
EOS_TOKEN_ID = tokenizer.eos_token_id  # End-of-sequence token ID for Mistral


class Sentence(NamedTuple):
    input_ids: torch.Tensor
    next_token_ids: torch.Tensor
    concepts: torch.Tensor


class NextTokenDataset(Dataset[Sentence]):
    def __init__(self, input_ids: torch.Tensor, concepts: torch.Tensor) -> None:
        super().__init__()
        self.input_ids = input_ids
        self.concepts = concepts

    def __len__(self) -> int:
        return self.num_sentences

    @property
    def num_sentences(self) -> int:
        return self.concepts.size(0)

    @property
    def context_window(self) -> int:
        return self.concepts.size(1)

    @property
    def num_concepts(self) -> int:
        return self.concepts.size(2)

    def __getitem__(self, idx: int) -> Sentence:
        input_ids = self.input_ids[idx]
        next_token_ids = torch.cat(
            [self.input_ids[idx][1:], torch.tensor([EOS_TOKEN_ID])],
        )
        concepts = self.concepts[idx]
        return Sentence(
            input_ids=input_ids,
            next_token_ids=next_token_ids,
            concepts=concepts,
        )


# %% [markdown]
# ## 2 - Define plotting function

# %%
def plot_learning_curves(
    training_losses: list[float],
    validation_losses: list[float],
    best_epoch: int,
    tg_token: str | None = None,
    tg_chat_id: str | None = None,
) -> None:
    """Plot and optionally send learning curves via Telegram."""
    fig, ax = plt.subplots()

    ax.plot(range(1, len(training_losses) + 1), training_losses, label="Training")
    ax.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation")
    ax.scatter(best_epoch + 1, training_losses[best_epoch])
    ax.scatter(best_epoch + 1, validation_losses[best_epoch])

    # Add best epoch as a minor tick with label
    current_minor_ticks = list(ax.get_xticks(minor=True))
    if best_epoch + 1 not in current_minor_ticks:
        current_minor_ticks.append(best_epoch + 1)
        current_minor_ticks.sort()
    ax.set_xticks(current_minor_ticks, minor=True)

    # Set label only for the best epoch tick
    minor_labels = [
        str(best_epoch + 1) if x == best_epoch + 1 else "" for x in current_minor_ticks
    ]
    ax.set_xticklabels(minor_labels, minor=True)

    # Check if best epoch label conflicts with major tick labels
    major_ticks = ax.get_xticks()

    # Estimate label width based on number of digits
    # Rough estimate: each digit is about 2% of the x-axis range
    digit_width = 0.02
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    best_label_width = len(str(int(best_epoch + 1))) * digit_width * x_range

    has_conflict = False
    for tick in major_ticks:
        tick_label_width = len(str(int(tick))) * digit_width * x_range
        min_distance = (best_label_width + tick_label_width) / 2
        if abs(best_epoch + 1 - tick) < min_distance:
            has_conflict = True
            break

    # Only move label inside plot if there's a conflict
    if has_conflict:
        ax.tick_params(axis="x", which="minor", pad=-20)

    # ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.legend()

    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=300)
    if tg_token and tg_chat_id:
        try:
            send_telegram_photo(
                token=tg_token,
                chat_id=tg_chat_id,
                photo_path="learning_curves.png",
                caption=r"SAE Next\-token classifier learning curves",
            )
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Failed to send Telegram photo: {e}")

    plt.show()


# %% [markdown] _cell_guid="c8dec0c1-cfb6-4fe8-bdbd-982dc0735f16" _uuid="20f7b8a2-7fa5-46b8-9ada-48535af1b2a3" jupyter={"outputs_hidden": false}
# ## 3 - Define Classifier

# %% _cell_guid="492d9dde-8a86-484f-9dbb-b20fe85c7f0b" _uuid="67f185da-6a4b-4d11-90e3-e40a06f13696" jupyter={"outputs_hidden": false}
LEARNING_RATE = 5e-4
NUM_EPOCHS = 20  # Remember it takes about 8m 35s per epoch on Kaggle
WARMUP_EPOCHS = 10
BATCH_SIZE = 128

# %% _cell_guid="8104a9e1-e3de-4e73-ab54-847876a1dd26" _uuid="3a184c6d-c6eb-4142-acaa-42c6bab5b327" jupyter={"outputs_hidden": false}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(NUM_CONCEPTS, VOCAB_SIZE),
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

# %% [markdown]
# ## 4 - Train classifier

# %% _cell_guid="2900cf5a-c558-47a1-a1c3-cf0f00ba9e14" _uuid="b4cf40ec-1516-4e5d-9e0e-9d086770f859" jupyter={"outputs_hidden": false}
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

v.on_train_begin(NUM_EPOCHS)
training_losses: list[float] = []
validation_losses: list[float] = []
best_val_loss = float("inf")
best_classifier_state = classifier.state_dict()
best_epoch = -1
for epoch in v.wrap_epoch_iterator(range(NUM_EPOCHS)):
    # v.on_epoch_begin(epoch)

    epoch_loss = 0.0
    n_batches = 0
    random.shuffle(train_chunks_paths)
    classifier.train()
    for chunk_idx, chunk_path in enumerate(train_chunks_paths):
        print(
            f"  Training on chunk {chunk_idx + 1}/{len(train_chunks_paths)}",
            end="\r",
        )
        chunk_input_ids, chunk_concepts = torch.load(chunk_path).values()

        chunk_dataset = NextTokenDataset(
            input_ids=chunk_input_ids,
            concepts=chunk_concepts,
        )
        chunk_dataloader = DataLoader(
            chunk_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        n_batches += len(chunk_dataloader)

        for batch in chunk_dataloader:
            concepts_batch = batch.concepts.to(device)
            next_token_ids_batch = batch.next_token_ids.to(device)

            logits = classifier(concepts_batch)
            # Masking
            # TODO: do not mast last token so that network can learn to predict EOS
            mask = next_token_ids_batch != EOS_TOKEN_ID
            loss = criterion(
                logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)],
                next_token_ids_batch.reshape(-1)[mask.reshape(-1)],
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

    epoch_loss /= n_batches
    training_losses.append(epoch_loss)

    val_loss = 0.0
    n_val_batches = 0
    random.shuffle(val_chunks_paths)
    classifier.eval()
    for chunk_idx, chunk_path in enumerate(val_chunks_paths):
        print(
            f"  Evaluating on chunk {chunk_idx + 1}/{len(val_chunks_paths)}",
            end="\r",
        )
        chunk_input_ids, chunk_concepts = torch.load(chunk_path).values()

        chunk_dataset = NextTokenDataset(
            input_ids=chunk_input_ids,
            concepts=chunk_concepts,
        )
        chunk_dataloader = DataLoader(
            chunk_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        n_val_batches += len(chunk_dataloader)

        with torch.inference_mode():
            for batch in chunk_dataloader:
                concepts_batch = batch.concepts.to(device)
                next_token_ids_batch = batch.next_token_ids.to(device)

                logits = classifier(concepts_batch)
                # Masking
                # TODO: do not mast last token so that network can learn to predict EOS
                mask = next_token_ids_batch != EOS_TOKEN_ID
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)],
                    next_token_ids_batch.reshape(-1)[mask.reshape(-1)],
                )

                val_loss += loss.item()

    val_loss /= n_val_batches
    validation_losses.append(val_loss)

    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = val_loss
        best_classifier_state = classifier.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": best_classifier_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_losses": training_losses,
                "validation_losses": validation_losses,
            },
            "next_token_checkpoint.pt",
        )
        print(f"Saved checkpoint at epoch {epoch + 1}")

    scheduler.step()
    if epoch % 10 == 0 and epoch != 0:
        plot_learning_curves(
            training_losses,
            validation_losses,
            best_epoch,
            tg_token=TG_TOKEN,
            tg_chat_id=TG_CHAT_ID,
        )
    v.on_epoch_end(epoch, train_loss=epoch_loss, val_loss=val_loss)

v.on_train_end()

torch.save(best_classifier_state, "next_token_classifier.pt")

# %%
plot_learning_curves(
    training_losses,
    validation_losses,
    best_epoch,
    tg_token=TG_TOKEN,
    tg_chat_id=TG_CHAT_ID,
)

# %%
import json

losses = {
    "training_losses": training_losses,
    "validation_losses": validation_losses,
}
with Path("losses.json").open("w") as f:
    json.dump(losses, f)

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
# # Classifier training on SAE concepts for next-token prediction
#
# This notebook trains a classifier head from the concepts extracted by a SAE from precomputed Mistral embeddings for the SST2 dataset to predict the next token.
#
# The output of this notebook is the state dict of the classifier head with the lowest validation loss, saved as `clf_sae_concepts_to_vocab/best_classifier_state.pt`, along with the training and validation losses saved as `clf_sae_concepts_to_vocab/losses.json` and the learning curves plot `clf_sae_concepts_to_vocab/losses.png`.

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
# #### Import libraries

# %% _cell_guid="eda13012-8054-4391-8b14-c0e5dc6a8076" _uuid="7dc3be8b-265a-4ae0-a41e-5d059fd3153f" jupyter={"outputs_hidden": false}
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

from lcblm.sae_utils import SparseAE, TopK
from lcblm.utils.data import NextTokenDataset, Sentence, typed_dataloader
from lcblm.utils.memory import free_gpu_memory
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

# %% [markdown] _cell_guid="0c74aef7-52af-420a-9b43-f1dddf123b9a" _uuid="10c3d1ab-2c21-4691-b59e-68e41614dd0d" jupyter={"outputs_hidden": false}
# #### Setup Telegram link

# %% _cell_guid="5d56351d-8fe2-4eca-8644-5f9fff5afa3e" _uuid="ef487cc3-bcc4-4268-840e-c5195b88ac18" jupyter={"outputs_hidden": false}
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
OUTPUT_PATH = Path("clf_head_sae_concepts")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# %% [markdown] _cell_guid="696e7b27-fbf2-498f-aabc-c4546a192f3d" _uuid="917f85df-7fab-4f3f-9a1e-a7eae45c23ad" jupyter={"outputs_hidden": false}
# ## 1 - Load Data

# %% _cell_guid="3a2f8a0d-2755-4965-b66a-bca0dec41b2b" _uuid="d0d26f22-93a2-472c-92a7-da8dc71f45eb" jupyter={"outputs_hidden": false}
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
    split: NextTokenDataset(
        input_ids=data[split]["input_ids"],
        attention_mask=data[split]["attention_masks"],
        embeddings=data[split]["embeddings"].float(),
        eos_token_id=EOS_TOKEN_ID,
    )
    for split in SPLITS
}

# %% [markdown]
# #### 1.3 - Load SAE

# %%
TOP_K = 128  # TODO: read k from file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sae_state_dict = torch.load(
    "/kaggle/input/sae-on-mistral-embeddings/pytorch/default/1/best_sae_state.pt",
)
in_dimension: int = sae_state_dict["lin_encoder.weight"].shape[1]
latent_dim: int = sae_state_dict["lin_encoder.weight"].shape[0]
sae = SparseAE(
    input_dim=in_dimension,
    latent_dim=latent_dim,
    activation=TopK(k=TOP_K),
)
sae.load_state_dict(sae_state_dict)
sae.to(device)
print(sae)
sae.eval()

free_gpu_memory(objects=["sae_state_dict"])

# %% [markdown]
# ## 2 - Define classifier and training parameters

# %% _cell_guid="492d9dde-8a86-484f-9dbb-b20fe85c7f0b" _uuid="67f185da-6a4b-4d11-90e3-e40a06f13696" jupyter={"outputs_hidden": false}
LEARNING_RATE = 5e-4
NUM_EPOCHS = 20  # 6m 30s per epoch
WARMUP_EPOCHS = 10
BATCH_SIZE = 64

# %% _cell_guid="8104a9e1-e3de-4e73-ab54-847876a1dd26" _uuid="3a184c6d-c6eb-4142-acaa-42c6bab5b327" jupyter={"outputs_hidden": false}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(latent_dim, VOCAB_SIZE),
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
# ## 4 - Train classifier

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
    msg="Starting training of *classifier head from concepts*",
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

        concepts_batch = sae(embeddings_batch).latents

        logits = classifier(concepts_batch)
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

            concepts_batch = sae(embeddings_batch).latents

            logits = classifier(concepts_batch)
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
            title="Classifier SAE Concepts Learning Curves",
            best_epoch=best_epoch,
        ) as (fig, ax):
            ax.set_ylabel("CE Loss")
            fig.savefig(LEARNING_CURVES_PATH, dpi=300)
            if TG_TOKEN is not None and TG_CHAT_ID is not None:
                msg_id = send_learning_curves_to_telegram(
                    image_path=LEARNING_CURVES_PATH,
                    tg_token=TG_TOKEN,
                    tg_chat_id=TG_CHAT_ID,
                    caption="SAE Concepts Classifier intermediate curves",
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
    title="Classifier SAE Concepts Learning Curves",
    best_epoch=best_epoch,
) as (fig, ax):
    ax.set_ylabel("CE Loss")
    fig.savefig(LEARNING_CURVES_PATH, dpi=300)
    if TG_TOKEN is not None and TG_CHAT_ID is not None:
        msg_id = send_learning_curves_to_telegram(
            image_path=LEARNING_CURVES_PATH,
            tg_token=TG_TOKEN,
            tg_chat_id=TG_CHAT_ID,
            caption="SAE Concepts Classifier final curves",
            msg_id=msg_id,
        )
    plt.show()

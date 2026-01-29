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
# # Extracting sst2 embeddings from Mistral model
#
# This notebook extracts embeddings from the Mistral model for the SST-2 dataset. The embeddings are obtained from the
# last layer of the LLM after tokenizing the sentences using Mistral's tokenizer.
#
# The output of this notebook consists of two files: the embeddings for the training set and the embeddings for the validation set. Each embedding is saved in a `.pt` file, which contains a dictionary with the following keys:
# - `input_ids`: The tokenized input IDs for each sentence.
# - `attention_masks`: The attention masks corresponding to the input IDs.
# - `embeddings`: The extracted embeddings from the last layer of the Mistral model.

# %% [markdown]
# ## 0 - Environment Setup

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
# ### 0.1 - Import libraries

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import os
from dataclasses import dataclass, field

import torch
from better_kaggle_secrets import UserSecretsClient
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from trainvox import send_telegram_message
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


# %% [markdown]
# ### 0.2 - Setup Telegram

# %%
user_secrets = UserSecretsClient()
TG_TOKEN = user_secrets.get_secret("TELEGRAM_TOKEN")
TG_CHAT_ID = user_secrets.get_secret("TELEGRAM_CHAT_ID")


# %% [markdown]
# ## 1 - Create dataset

# %% [markdown]
# #### Embedding extraction configuration


# %%
@dataclass
class ExtractionConfig:
    model_name: str
    dataset: str
    batch_size: int
    max_length: int  # max length of tokenized sequence
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    )


config = ExtractionConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    dataset="SetFit/sst2",
    batch_size=64,
    max_length=256,
)

# %% [markdown]
# #### Load tokenizer

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # don't know why this is needed

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# #### Load LLM model

# %%
llm = AutoModel.from_pretrained(
    config.model_name,
    dtype=torch.float16,  # With float32 goes out of memory
    device_map="auto",  # Automatically splits model across GPUs
)

llm.eval()
for p in llm.parameters():
    p.requires_grad = False

# %% [markdown]
# #### Load dataset and relevant tokenization setup

# %%
datasets: DatasetDict = load_dataset(config.dataset)  # pyright: ignore[reportAssignmentType]
datasets.pop("test", None)  # we don't need the test set
for split, dataset in datasets.items():
    print(f"{split} samples: {len(dataset)}")

remove_columns = ["text", "label_text"]


def tokenize_function(examples):  # noqa: ANN001, ANN201
    return tokenizer(
        examples["text"],
        padding=True,
        truncation=True,
        max_length=config.max_length,
    )


class ExtractionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, encodings) -> None:  # noqa: ANN001
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(self.encodings[key][idx])
            for key in self.encodings.features
        }
        return item


# %% [markdown]
# #### Extract data

# %%
if TG_TOKEN is not None and TG_CHAT_ID is not None:
    send_telegram_message(
        "🚀 Starting embedding extraction process.",
        token=TG_TOKEN,
        chat_id=TG_CHAT_ID,
    )

for split, dataset in datasets.items():
    print(f"\n--- Processing split: {split} ---")

    # Tokenize dataset
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=len(dataset),
        remove_columns=remove_columns,
    )
    # Create PyTorch Dataset and DataLoader
    extraction_dataset = ExtractionDataset(encoded_dataset)
    extraction_loader = DataLoader(
        extraction_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Shuffling is not needed for feature extraction
    )

    all_input_ids: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_embeddings: list[torch.Tensor] = []

    # Pass data through Mistral to get embeddings
    with torch.inference_mode():
        for batch in tqdm(
            extraction_loader,
            desc=f"Extracting {split} embeddings",
            unit="batch",
        ):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            outputs = llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeddings = outputs.last_hidden_state

            all_input_ids.append(input_ids.cpu())
            all_attention_masks.append(attention_mask.cpu())
            all_embeddings.append(embeddings.cpu())

    # Stack everything
    all_input_ids_tensor = torch.cat(all_input_ids, dim=0)
    all_attention_masks_tensor = torch.cat(all_attention_masks, dim=0)
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

    # Save to disk
    torch.save(
        {
            "input_ids": all_input_ids_tensor,
            "attention_masks": all_attention_masks_tensor,
            "embeddings": all_embeddings_tensor,
        },
        f"extracted_features_{split}.pt",
    )

if TG_TOKEN is not None and TG_CHAT_ID is not None:
    send_telegram_message(
        "✅ Embedding extraction completed successfully.",
        token=TG_TOKEN,
        chat_id=TG_CHAT_ID,
    )

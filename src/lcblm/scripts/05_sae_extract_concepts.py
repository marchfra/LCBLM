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
# # SAE concept extraction for sst2's Mistral embeddings
#
# Starting from sst2 I need to:
# 1. tokenize the sentences with Mistral's tokenizer
# 2. pass the tokenized sentences through Mistral to get the embeddings at the last layer
#     - in the future, get the embeddings from various layers
# 3. train a SAE to map from the embedding of the last layer to concept space
#     - in the future, train a SAE to map from various layers to concept space
# 4. extract the concept representations for each token in the sentence
# 5. train a linear classifier to predict the *next* token id from the concept representation of the *current* token
#
# The focus of this notebook is step 4.
#
# Since the train set tensor is too large to fit in memory, we will extract the concept representations in chunks and save them to disk.
#
# This notebook will output `n_chunks` files for each dataset (train, validation, test), which must be manually downloaded after each chunk is processed, as per the instructions in the last cell.
#
# The output files will be named as follows:
# - `f"{extractor.output_dir}/latent_chunks_{chunk_idx:4d}.pt"`
#
# to distinguish the various splits, the files must be manually renamed.

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
    raise SystemExit("This notebook is intended to run on Kaggle only.")

# %% [markdown]
# ### 0.1 - Import libraries

# %%
import gc
import json
from pathlib import Path
from typing import NamedTuple

import torch
from sae_utils import SparseAE, TopK
from torch.utils.data import Dataset

# torch.manual_seed(15972821823142484745)
torch.manual_seed(3742)

# %% [markdown]
# ## 1 - Load Data and Pretrained SAE

# %%
PRETRAINED_SAE_PATH = (
    Path("/kaggle/input")
    / "sae-on-mistral-embeddings"
    / "pytorch/default/1"
    / "sae_on_mistral_sst2-200epochs.pt"
)

if not PRETRAINED_SAE_PATH.exists():
    raise FileNotFoundError("Pretrained SAE model file is missing.")
else:
    print("Found SAE at specified location")

# %%
DATA_PATH = Path("/kaggle/input/sst2-mistral-embeddings")
TRAIN_PATH = DATA_PATH / "extracted_features_train.pt"
VAL_PATH = DATA_PATH / "extracted_features_validation.pt"
TEST_PATH = DATA_PATH / "extracted_features_test.pt"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data path {DATA_PATH} does not exist.")


if TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists():
    train_data = torch.load(TRAIN_PATH)
    val_data = torch.load(VAL_PATH)
    test_data = torch.load(TEST_PATH)
else:
    raise FileNotFoundError("Training, validation, or test data files are missing.")


# %%
class Item(NamedTuple):
    input_ids: torch.Tensor
    embeddings: torch.Tensor


class EmbeddingsDataset(Dataset[Item]):
    def __init__(self, input_ids: torch.Tensor, embeddings: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.embeddings = embeddings

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> Item:
        return Item(self.input_ids[idx], self.embeddings[idx])



# %%
train_dataset = EmbeddingsDataset(train_data["input_ids"], train_data["embeddings"])
val_dataset = EmbeddingsDataset(val_data["input_ids"], val_data["embeddings"])
test_dataset = EmbeddingsDataset(test_data["input_ids"], test_data["embeddings"])

del train_data, val_data, test_data
gc.collect()

# %%
LATENT_DIM_FACTOR = 4
TOP_K = 128

_sae = SparseAE(
    input_dim=train_dataset.embeddings.shape[-1],
    latent_dim_factor=LATENT_DIM_FACTOR,
    activation=TopK(k=TOP_K),
)
_sae.load_state_dict(torch.load(PRETRAINED_SAE_PATH))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sae = torch.nn.DataParallel(_sae).to(device)

del _sae
gc.collect()


# %% [markdown]
# ## 2 - Compute latent space from Mistral embeddings in chunks

# %%
class ChunkedLatentExtractor:
    def __init__(
        self,
        sae_model: torch.nn.DataParallel[SparseAE],
        dataset: EmbeddingsDataset,
        output_dir: str | Path = "latent_chunks",
        samples_per_chunk: int | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        """Init method.

        Args:
            sae_model: Your trained SAE model
            dataset: Your dataset (train/val/test)
            output_dir: Directory to save chunks
            samples_per_chunk: Number of samples per chunk (auto-calculate if None)
            latent_dim: Dimension of latent space
            device: Device to run on

        """
        self.sae_model = sae_model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.latent_dim = sae_model.module.latent_dim
        self.device = device

        # Calculate chunk size based on available disk space
        if samples_per_chunk is None:
            self.samples_per_chunk = self._calculate_chunk_size()
        else:
            self.samples_per_chunk = samples_per_chunk

        self.checkpoint_file = self.output_dir / "checkpoint.json"

    def _calculate_chunk_size(self, safety_margin: float = 0.5) -> int:
        """Calculate how many samples can fit in available disk space.

        Assumes float32 (4 bytes per value).
        """
        available_disk_gb = 19.5 * safety_margin  # Use 50% of available space
        bytes_per_sample = (
            self.dataset[0].embeddings.shape[:-1].numel() * self.latent_dim * 4
        )  # float32
        samples = int((available_disk_gb * 1024**3) / bytes_per_sample)
        print(f"Calculated chunk size: {samples:,} samples")
        print(
            f"Estimated chunk size on disk: {samples * bytes_per_sample / 1024**3:.2f} "
            f"GiB",
        )
        return samples

    def _save_checkpoint(self, chunk_idx: int, total_processed: int) -> None:
        """Save progress checkpoint."""
        checkpoint = {
            "last_chunk_idx": chunk_idx,
            "total_processed": total_processed,
            "samples_per_chunk": self.samples_per_chunk,
        }
        with self.checkpoint_file.open("w") as f:
            json.dump(checkpoint, f)

    def _load_checkpoint(self) -> dict[str, int] | None:
        """Load progress checkpoint."""
        if self.checkpoint_file.exists():
            with self.checkpoint_file.open("r") as f:
                return json.load(f)
        return None

    def extract_chunk(
        self,
        start_idx: int,
        end_idx: int,
        chunk_idx: int,
        batch_size: int = 32,
    ) -> tuple[Path, float]:
        """Extract latents for a specific chunk of the dataset.

        Args:
            start_idx: Starting index in dataset
            end_idx: Ending index in dataset
            chunk_idx: Chunk number for saving
            batch_size: Batch size for processing

        """
        print(f"\n{'=' * 60}")
        print(f"Processing Chunk {chunk_idx}")
        print(f"Samples: {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} total)")
        print(f"{'=' * 60}")

        self.sae_model.eval()
        input_ids_list = []
        latents_list = []

        # # ! Inference mode should be even faster than no_grad, however things may break.
        # # ! If they do, simply go back to torch.no_grad()
        with torch.inference_mode():
            for i in range(start_idx, end_idx, batch_size):
                batch_end = min(i + batch_size, end_idx)

                # Get batch from dataset
                batch = self.dataset[i:batch_end]  # pyright: ignore[reportArgumentType]
                embeddings = batch.embeddings.to(self.device)

                # Extract latents
                latents = self.sae_model(embeddings).latents

                # Move to CPU immediately to free GPU memory
                input_ids_list.append(batch.input_ids)
                latents_list.append(latents.cpu())

                # Clear GPU cache periodically
                if (i - start_idx) % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    print(
                        f"  Processed {i - start_idx:,}/{end_idx - start_idx:,} "
                        f"samples",
                        end="\r",
                    )

        print(f"  Processed {end_idx - start_idx:,}/{end_idx - start_idx:,} samples")

        # Concatenate all batches
        print("  Concatenating batches...")
        chunk_input_ids = torch.cat(input_ids_list, dim=0)
        chunk_latents = torch.cat(latents_list, dim=0)

        # Save chunk
        chunk_file = self.output_dir / f"latents_chunk_{chunk_idx:04d}.pt"
        print(f"  Saving to {chunk_file}...")
        torch.save({"input_ids": chunk_input_ids, "latents": chunk_latents}, chunk_file)

        # Get file size
        file_size_gb = chunk_file.stat().st_size / 1024**3
        print(f"  Saved! File size: {file_size_gb:.2f} GiB")

        # Save checkpoint
        self._save_checkpoint(chunk_idx, end_idx)

        return chunk_file, file_size_gb

    def extract_all(
        self,
        batch_size: int = 32,
        *,
        start_from_checkpoint: bool = True,
    ) -> None:
        """Extract all latents in chunks.

        Args:
            batch_size: Batch size for processing
            start_from_checkpoint: Whether to resume from checkpoint

        """
        total_samples = len(self.dataset)

        # Check for checkpoint
        start_idx = 0
        start_chunk_idx = 0

        if start_from_checkpoint:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                start_idx = checkpoint["total_processed"]
                start_chunk_idx = checkpoint["last_chunk_idx"] + 1
                print(
                    f"Resuming from checkpoint: "
                    f"{start_idx:,}/{total_samples:,} samples",
                )

        # Calculate chunks
        chunks = []
        for i in range(start_idx, total_samples, self.samples_per_chunk):
            end = min(i + self.samples_per_chunk, total_samples)
            chunks.append((i, end))

        print(f"Total samples: {total_samples:,}")
        print(f"Samples per chunk: {self.samples_per_chunk:,}")
        print(f"Number of chunks to process: {len(chunks)}")
        print("\nStarting extraction...")

        # Process chunks
        for chunk_num, (start, end) in enumerate(chunks, start=start_chunk_idx):
            self.extract_chunk(start, end, chunk_num, batch_size)

            print(f"\n✓ Chunk {chunk_num} complete!")
            print(
                f"  Progress: {end:,}/{total_samples:,} samples "
                f"({100 * end / total_samples:.1f}%)",
            )

            if chunk_num < len(chunks) + start_chunk_idx - 1:
                print(f"\n{'=' * 60}")
                print("ACTION REQUIRED:")
                print("1. Download the chunk file(s) from Kaggle")
                print("2. Delete the chunk file(s) from Kaggle to free disk space")
                print("3. Re-run this script to continue")
                print(f"{'=' * 60}\n")
                break
        else:
            print(f"\n{'=' * 60}")
            print("✓ ALL CHUNKS EXTRACTED!")
            print(f"{'=' * 60}\n")


# %%
# Then repeat for validation and test datasets
extractor = ChunkedLatentExtractor(
    sae_model=sae,
    dataset=train_dataset,
    output_dir="latent_chunks_train",
    device="cpu",
)

for file in extractor.output_dir.glob("*.pt"):
    print(f"Deleting file: {file}")
    file.unlink()

extractor.extract_all(batch_size=512)

"""Data loading and preprocessing for the VAEE vs SparseAE experiment.

Loads pre-extracted LLM embeddings from .pt files, applies sentence-level
subsampling, fits a StandardScaler on the training token embeddings, and
returns NextTokenDataset objects with normalised embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from sklearn.preprocessing import StandardScaler

from lcblm.utils.data import NextTokenDataset

from .exp_config import DatasetConfig

if TYPE_CHECKING:
    from collections.abc import Callable


def load_sst2_mistral(
    ds_cfg: DatasetConfig,
) -> tuple[NextTokenDataset, NextTokenDataset, StandardScaler]:
    """Load and normalise pre-extracted Mistral SST2 embeddings.

    Fits a StandardScaler on the training token embeddings (padding excluded),
    applies it to both splits, then wraps the result in NextTokenDataset objects.

    Args:
        ds_cfg: Dataset configuration, including path, input_dim, and n_samples.

    Returns:
        train_dataset, val_dataset, scaler

    """
    path = Path(ds_cfg.embeddings_path)
    splits = ["train", "validation"]

    raw: dict[str, dict[str, torch.Tensor]] = {}
    for split in splits:
        pt_path = path / f"extracted_features_{split}.pt"
        if not pt_path.exists():
            msg = f"Embeddings file not found: {pt_path}"
            raise FileNotFoundError(msg)
        raw[split] = torch.load(pt_path, weights_only=True)

    # Sentence-level subsampling
    for split in splits:
        n = ds_cfg.n_samples
        if n != -1:
            for key in raw[split]:
                raw[split][key] = raw[split][key][:n]

    train_emb = raw["train"]["embeddings"].float()  # (S, L, D)
    train_mask = raw["train"]["attention_masks"].bool()  # (S, L)
    val_emb = raw["validation"]["embeddings"].float()

    # Fit scaler on flattened training tokens (padding excluded)
    flat_train = train_emb[train_mask].numpy()  # (N_tokens, D)
    scaler = StandardScaler()
    scaler.fit(flat_train)

    # Apply scaler to all positions (padding is masked out during training)
    s_tr, l_tr, d = train_emb.shape
    train_norm = torch.from_numpy(
        scaler.transform(train_emb.reshape(-1, d).numpy()).astype("float32"),
    ).reshape(s_tr, l_tr, d)

    s_va, l_va, _ = val_emb.shape
    val_norm = torch.from_numpy(
        scaler.transform(val_emb.reshape(-1, d).numpy()).astype("float32"),
    ).reshape(s_va, l_va, d)

    train_dataset = NextTokenDataset(
        input_ids=raw["train"]["input_ids"],
        attention_mask=raw["train"]["attention_masks"],
        embeddings=train_norm,
        eos_token_id=ds_cfg.eos_token_id,
    )
    val_dataset = NextTokenDataset(
        input_ids=raw["validation"]["input_ids"],
        attention_mask=raw["validation"]["attention_masks"],
        embeddings=val_norm,
        eos_token_id=ds_cfg.eos_token_id,
    )

    return train_dataset, val_dataset, scaler


# Registry mapping dataset name → (default DatasetConfig, loader function).
# The embeddings_path in the default config is a placeholder; it must be
# overridden via the TOML config before calling the loader.
DATASET_REGISTRY: dict[
    str,
    tuple[DatasetConfig, Callable],
] = {
    "sst2_mistral": (
        DatasetConfig(
            name="sst2_mistral",
            embeddings_path="",
            input_dim=4096,
            eos_token_id=2,
        ),
        load_sst2_mistral,
    ),
}

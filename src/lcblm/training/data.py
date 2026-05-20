"""Load pre-extracted LLM embeddings for concept model training.

Loads pre-extracted LLM embeddings from .pt files, fits a StandardScaler on
training tokens, and returns normalised NextTokenDataset objects.

Expected keys in each extracted_features_*.pt file:
  - input_ids:       (n_sentences, seq_len)  int
  - attention_masks: (n_sentences, seq_len)  bool
  - embeddings:      (n_sentences, seq_len, input_dim)  float
  - word_ids:        (n_sentences, seq_len)  int, -1 for special tokens  [optional]
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler

from lcblm.utils.data import FlatTensorDataset, NextTokenDataset


def _normalise(emb: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    s, l, d = emb.shape  # noqa: E741
    return torch.from_numpy(
        scaler.transform(emb.reshape(-1, d).numpy()).astype("float32"),
    ).reshape(s, l, d)


def _make_dataset(
    raw: dict[str, torch.Tensor],
    norm_emb: torch.Tensor,
    eos_token_id: int,
) -> NextTokenDataset:
    return NextTokenDataset(
        input_ids=raw["input_ids"],
        attention_mask=raw["attention_masks"],
        embeddings=norm_emb,
        eos_token_id=eos_token_id,
        word_ids=raw.get("word_ids"),
    )


def load_embeddings(
    embeddings_path: str | Path,
    eos_token_id: int,
    n_samples: int = -1,
) -> tuple[NextTokenDataset, NextTokenDataset, StandardScaler]:
    """Load and normalise pre-extracted embeddings for both splits.

    Fits a StandardScaler on training token embeddings (padding excluded) and
    applies it to both splits.

    Args:
        embeddings_path: Directory containing extracted_features_train.pt and
            extracted_features_validation.pt.
        eos_token_id: EOS token ID for NextTokenDataset.
        n_samples: Number of sentences to load per split. -1 means all.

    Returns:
        train_dataset, val_dataset, scaler

    """
    path = Path(embeddings_path)
    raw: dict[str, dict[str, torch.Tensor]] = {}
    for split in ("train", "validation"):
        pt = path / f"extracted_features_{split}.pt"
        if not pt.exists():
            msg = f"Embeddings file not found: {pt}"
            raise FileNotFoundError(msg)
        raw[split] = torch.load(pt, weights_only=True)
        if n_samples != -1:
            for key in raw[split]:
                raw[split][key] = raw[split][key][:n_samples]

    train_emb = raw["train"]["embeddings"].float()
    scaler = StandardScaler()
    scaler.fit(train_emb[raw["train"]["attention_masks"].bool()].numpy())

    train_ds = _make_dataset(raw["train"], _normalise(train_emb, scaler), eos_token_id)
    val_ds = _make_dataset(
        raw["validation"],
        _normalise(raw["validation"]["embeddings"].float(), scaler),
        eos_token_id,
    )
    return train_ds, val_ds, scaler


def load_split(
    embeddings_path: str | Path,
    split: str,
    scaler: StandardScaler,
    eos_token_id: int,
    n_samples: int = -1,
) -> NextTokenDataset:
    """Load a single split, normalising with a pre-fitted scaler.

    Args:
        embeddings_path: Directory containing extracted_features_*.pt.
        split: "train" or "val".
        scaler: Fitted StandardScaler (from load_embeddings).
        eos_token_id: EOS token ID for NextTokenDataset.
        n_samples: Number of sentences to load. -1 means all.

    Returns:
        Normalised NextTokenDataset for the requested split.

    """
    path = Path(embeddings_path)
    fname = "validation" if split == "val" else split
    raw = torch.load(path / f"extracted_features_{fname}.pt", weights_only=True)
    if n_samples != -1:
        for key in raw:
            raw[key] = raw[key][:n_samples]

    return _make_dataset(
        raw,
        _normalise(raw["embeddings"].float(), scaler),
        eos_token_id,
    )


def flatten_token_dataset(ds: NextTokenDataset) -> FlatTensorDataset:
    """Materialise the attended-to tokens of a NextTokenDataset as a flat dataset.

    Each unpadded token becomes one (input_dim,) sample. Used to feed
    token-embedding data into the dict-learning training loops, which consume
    flat ``FlatTensorDataset`` instances uniformly with image/synthetic data.
    """
    return FlatTensorDataset(ds.embeddings[ds.attention_mask])


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: Path) -> StandardScaler:
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301

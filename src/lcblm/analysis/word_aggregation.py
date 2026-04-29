"""Token-to-word alpha aggregation using HuggingFace word_ids.

HF fast tokenizers expose word_ids() on BatchEncoding objects, mapping each
sub-word token to the original word index (None for special tokens). This
module aggregates per-token activation values to per-word values by grouping
consecutive tokens that share the same (sentence_idx, word_id) pair.

Usage requires that word_ids were stored alongside the embeddings at extraction
time (add a ``word_ids`` key to the extracted_features_*.pt files). Currently
those files do not include word_ids; this module is ready for when they do.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def aggregate_to_words(  # noqa: PLR0913
    alpha: np.ndarray,
    token_ids: np.ndarray,
    sentence_indices: np.ndarray,
    positions: np.ndarray,
    word_ids: np.ndarray,
    method: Literal["mean", "max", "sum"] = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-token activation values to per-word activation values.

    Groups tokens by (sentence_idx, word_id) and reduces alpha across sub-word
    tokens in each group. The representative token_id and position for each word
    are taken from the first sub-word token (lowest position within the group).

    Special tokens (word_id == -1) are excluded from the output.

    Args:
        alpha: Per-token activations, shape (N_tokens, n_concepts).
        token_ids: Token ID for each token, shape (N_tokens,).
        sentence_indices: Sentence index for each token, shape (N_tokens,).
        positions: Position within the padded sentence for each token, shape
            (N_tokens,).
        word_ids: Word index for each token, shape (N_tokens,). Use -1 for special
            tokens (BOS, EOS, PAD) that should be excluded.
        method: Aggregation function across sub-word tokens. "mean" (default), "max",
            or "sum".

    Returns:
        word_alpha: Aggregated activations, shape (N_words, n_concepts).
        word_token_ids: Token ID of the first sub-word token of each word.
        word_sentence_indices: Sentence index for each word.
        word_positions: Position of the first sub-word token of each word.

    """
    real_mask = word_ids != -1
    alpha = alpha[real_mask]
    token_ids = token_ids[real_mask]
    sentence_indices = sentence_indices[real_mask]
    positions = positions[real_mask]
    word_ids = word_ids[real_mask]

    # Unique (sentence, word) pairs, preserving first-occurrence order
    keys = np.stack([sentence_indices, word_ids], axis=1)
    _, first_occurrence, inverse = np.unique(
        keys,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    n_words = len(first_occurrence)
    n_concepts = alpha.shape[1]

    if method == "max":
        word_alpha = np.full((n_words, n_concepts), -np.inf, dtype=np.float32)
        np.maximum.at(word_alpha, inverse, alpha)
    elif method == "sum":
        word_alpha = np.zeros((n_words, n_concepts), dtype=np.float32)
        np.add.at(word_alpha, inverse, alpha)
    else:
        word_alpha = np.zeros((n_words, n_concepts), dtype=np.float32)
        counts = np.zeros(n_words, dtype=np.int64)
        np.add.at(word_alpha, inverse, alpha)
        np.add.at(counts, inverse, 1)
        word_alpha /= counts[:, np.newaxis].clip(min=1)

    word_token_ids = token_ids[first_occurrence]
    word_sentence_indices = sentence_indices[first_occurrence]
    word_positions = positions[first_occurrence]

    return word_alpha, word_token_ids, word_sentence_indices, word_positions

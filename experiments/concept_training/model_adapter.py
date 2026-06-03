"""Model-agnostic adapter interface for concept model inference.

All concept models — VAEE, TopK SAE, L1 SAE — expose the same two-method
interface via ModelAdapter: encode returns per-token concept activations and
decode reconstructs the embedding. This lets build_cd.py and intervene.py
work without any model-type-specific logic.

Alpha semantics by model type:
  VAEE     — gate probability ∈ [0, 1]; active if alpha > 0.5 (default)
  TopK SAE — post-TopK activation value (positive float); active if value > 0
  L1 SAE   — ReLU activation value; active if value > 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

    from lcblm.baselines.vq_vae import VQVAE
    from lcblm.sae_utils.model import SparseAE
    from lcblm.utils.data import NextTokenDataset
    from lcblm.vaee.models import VAEE, VAEESharedEncoder


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class ModelAdapter(Protocol):
    n_concepts: int
    default_threshold: float

    def encode(self, x: Tensor) -> Tensor:
        """Map token embeddings to concept activations.

        Args:
            x: Token embeddings, shape (N_tokens, input_dim).

        Returns:
            Activations, shape (N_tokens, n_concepts).

        """
        ...

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct token embeddings from latent representations.

        Args:
            z: Latent tensor. For SparseAE: (N_tokens, n_concepts). For VAEE:
               (N_tokens, n_concepts, embedding_size) — the gated prototype tensor.

        Returns:
            Reconstructed embeddings, shape (N_tokens, input_dim).

        """
        ...

    def to(self, device: torch.device) -> ModelAdapter: ...


# ── Concrete adapters ─────────────────────────────────────────────────────────


class VAEEAdapter:
    """Adapter for VAEE. encode() returns gate probabilities (alpha ∈ [0, 1])."""

    def __init__(self, model: VAEE) -> None:
        self._model = model.eval()
        self.n_concepts = model.num_embeddings
        self.default_threshold = 0.5

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        return self._model(x).alpha

    @torch.inference_mode()
    def decode(self, z: Tensor) -> Tensor:
        return self._model.decode(z)

    def to(self, device: torch.device) -> VAEEAdapter:
        self._model = self._model.to(device)
        return self


class VAEESharedEncoderAdapter:
    """Adapter for VAEESharedEncoder. encode() returns gate probabilities (alpha ∈ [0, 1])."""  # noqa: E501

    def __init__(self, model: VAEESharedEncoder) -> None:
        self._model = model.eval()
        self.n_concepts = model.num_embeddings
        self.default_threshold = 0.5

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        return self._model(x).alpha

    @torch.inference_mode()
    def decode(self, z: Tensor) -> Tensor:
        return self._model.decode(z)

    def to(self, device: torch.device) -> VAEESharedEncoderAdapter:
        self._model = self._model.to(device)
        return self


class VQVAEAdapter:
    """Adapter for VQVAE. encode() returns a one-hot (B, num_codes) activation tensor."""  # noqa: E501

    def __init__(self, model: VQVAE) -> None:
        self._model = model.eval()
        self.n_concepts = model.num_codes
        self.default_threshold = 0.0

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        out = self._model(x)
        one_hot = torch.zeros(x.shape[0], self._model.num_codes, device=x.device)
        one_hot.scatter_(1, out.indices.unsqueeze(1), 1.0)
        return one_hot

    @torch.inference_mode()
    def decode(self, z: Tensor) -> Tensor:
        return self._model.decode(z)

    def to(self, device: torch.device) -> VQVAEAdapter:
        self._model = self._model.to(device)
        return self


class SparseAEAdapter:
    """Adapter for SparseAE (TopK or L1). encode() returns post-activation latents."""

    def __init__(self, model: SparseAE) -> None:
        self._model = model.eval()
        self.n_concepts = model.latent_dim
        self.default_threshold = 0.0

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        _, z = self._model.encode(x)
        return z

    @torch.inference_mode()
    def decode(self, z: Tensor) -> Tensor:
        return self._model.decode(z)

    def to(self, device: torch.device) -> SparseAEAdapter:
        self._model = self._model.to(device)
        return self


# ── Inference loop ────────────────────────────────────────────────────────────


def get_token_activations(
    adapter: ModelAdapter,
    dataset: NextTokenDataset,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the adapter over the full dataset and return per-token activations.

    Processes sentences in batches, flattening real tokens (non-padding) before
    passing them through the adapter. Order of returned flat arrays matches.

    Args:
        adapter: Model adapter with an encode() method.
        dataset: Normalised NextTokenDataset.
        batch_size: Number of sentences per inference batch.
        device: Inference device. Defaults to CUDA if available, else CPU.

    Returns:
        alpha: Activations, shape (N_tokens, n_concepts).
        token_ids: Flat token IDs, shape (N_tokens,).
        sentence_indices: Which sentence each token belongs to, shape (N_tokens,).
        positions: Position within the padded sentence, shape (N_tokens,).

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter.to(device)

    all_alpha: list[torch.Tensor] = []
    all_ids: list[torch.Tensor] = []
    all_sent: list[np.ndarray] = []
    all_pos: list[np.ndarray] = []

    n = dataset.num_sentences
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sentences = dataset[list(range(start, end))]

        emb = torch.stack([s.embeddings for s in sentences]).to(device)
        mask = torch.stack([s.attention_mask for s in sentences]).to(device)
        ids = torch.stack([s.input_ids for s in sentences])

        mask_cpu = mask.cpu()
        sent_idxs, positions = mask_cpu.nonzero(as_tuple=True)
        all_sent.append((sent_idxs + start).numpy())
        all_pos.append(positions.numpy())

        all_alpha.append(adapter.encode(emb[mask]).cpu())
        all_ids.append(ids[mask_cpu])

    return (
        torch.cat(all_alpha).numpy(),
        torch.cat(all_ids).numpy(),
        np.concatenate(all_sent),
        np.concatenate(all_pos),
    )

"""Unit tests for intervention metrics on hand-constructed models."""

from __future__ import annotations

import torch
from torch import nn

from lcblm.eval.intervention import (
    ablation_directions,
    evaluate_intervention,
    intervention_consistency,
    intervention_factor_recovery,
)
from lcblm.sae_utils.model import SparseAE
from lcblm.vaee.models import VAEE


def _linear_sae_with_known_atoms(atoms: torch.Tensor) -> SparseAE:
    """A linear SAE whose decoder columns are exactly ``atoms`` (rows of A).

    With encoder = A^T and decoder columns = atoms, the latents recover the
    coefficients and the ablation contribution of latent k is exactly
    coeff_k * atom_k — a known, input-independent direction.
    """
    dim = atoms.shape[0]
    model = SparseAE(
        input_dim=dim,
        latent_dim=dim,
        activation=nn.Identity(),
        tied_weights=False,
        use_tied_bias=False,
    )
    # atoms[k] is the k-th atom (unit row). encoder.weight (latent, input) = A;
    # decoder.weight (input, latent) has column k = atom_k = A[k] -> A^T.
    model._encoder.weight.data = atoms.clone()
    model._decoder.weight.data = atoms.t().clone()
    return model


def test_consistency_and_recovery_on_known_linear_model() -> None:
    torch.manual_seed(0)
    dim = 4
    # Orthonormal atom basis via QR of a random matrix.
    atoms, _ = torch.linalg.qr(torch.randn(dim, dim))
    model = _linear_sae_with_known_atoms(atoms)

    # Inputs are positive combinations of atoms so every latent is active (>0).
    coeffs = torch.rand(256, dim) + 0.5  # in [0.5, 1.5]
    x = coeffs @ atoms  # (N, dim); since latents = A x = coeffs

    mean_dir, resultant, valid = ablation_directions(
        model, "topk_sae", x, active_thresh=1e-6, min_active=8
    )
    assert bool(valid.all()), "every latent should be active on positive inputs"
    # A clean linear model gives a perfectly consistent ablation direction.
    assert torch.allclose(resultant[valid], torch.ones(dim), atol=1e-4)

    # Each mean direction must equal its atom (up to sign; coeffs > 0 -> +atom).
    cos = torch.nn.functional.normalize(mean_dir, dim=1) @ atoms.t()
    assert torch.allclose(cos.diag(), torch.ones(dim), atol=1e-4)

    cons = intervention_consistency(model, "topk_sae", x, active_thresh=1e-6)
    assert cons["mean_consistency"] > 0.999
    assert cons["n_concepts"] == dim

    fac = intervention_factor_recovery(
        model, "topk_sae", x, atoms, threshold=0.9, active_thresh=1e-6
    )
    assert fac["causal_matched_fraction"] == 1.0
    assert fac["causal_mean_cosine_sim"] > 0.999


def test_entangled_decoder_lowers_consistency() -> None:
    """If a latent's contribution depends on the input, consistency drops."""
    torch.manual_seed(1)
    dim = 4
    atoms = torch.eye(dim)
    model = _linear_sae_with_known_atoms(atoms)
    # Make latent 0 alternate sign across inputs so its ablation direction flips.
    x = torch.rand(256, dim) + 0.5
    x[::2, 0] = -(x[::2, 0])  # half the inputs drive latent 0 negative

    _, resultant, valid = ablation_directions(
        model, "topk_sae", x, active_thresh=-1e9, min_active=8
    )
    # Latent 0 fires both signs -> directions cancel -> low resultant; others stay 1.
    assert resultant[0] < 0.6
    assert resultant[1] > 0.99


def test_runs_on_vaee_smoke() -> None:
    """The gate-model path runs end to end and returns sane ranges."""
    torch.manual_seed(2)
    model = VAEE(
        input_dim=8, num_embeddings=6, embedding_size=4, sim_metric="inner_product"
    )
    model.eval()
    x = torch.randn(64, 8)
    atoms = torch.randn(6, 8)
    out = evaluate_intervention(model, "vaee", x, atoms, threshold=0.7, min_active=2)
    assert 0.0 <= out["mean_consistency"] <= 1.0 or out["n_concepts"] == 0
    assert 0.0 <= out["causal_matched_fraction"] <= 1.0

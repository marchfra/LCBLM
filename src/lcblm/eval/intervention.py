"""Intervention metrics for concept models (PLAN.md metrics 6 & 7, synthetic tier).

Two model-agnostic intervention metrics, defined by *ablating* a single concept
(setting its activation to zero) and measuring the induced change in the
reconstruction. Ablation is the cleanest model-agnostic "gate flip": it is
well-defined for VAEE gates, VAEE-SE gates and SAE latents alike.

For a concept ``k`` and an input ``x`` where ``k`` is active, the ablation
contribution is

    delta_k(x) = recon(x) - decode(activations with a_k := 0),

i.e. the part of the reconstruction that concept ``k`` is responsible for.

* **Metric 6 — intervention consistency** (``intervention_consistency``):
  the mean *resultant length* of the unit ablation directions across the inputs
  where ``k`` is active. A clean, controllable concept produces the *same*
  direction regardless of the input, so the unit vectors align and the resultant
  length approaches 1; an entangled / input-dependent concept gives directions
  that cancel, pushing it toward 0. This is the plan's "mean cosine similarity of
  gate-flip directions", computed directly in input space (no ResNet — that is
  only the image-space yardstick).

* **Metric 7 — causal factor recovery** (``intervention_factor_recovery``):
  the causal counterpart of prototype-based ``feature_recovery``. Each concept's
  mean ablation direction is Hungarian-matched to the ground-truth atoms; we
  report the matched fraction and mean cosine (reusing ``feature_recovery``) plus
  a per-concept *dominance ratio* (top atom alignment / second-best), which
  quantifies how cleanly an intervention maps to a single factor.

The metrics are deliberately checkpoint-free: they take an already-built model
plus its ``model_name`` and a batch of inputs, so they compose with both the
training loop and the offline reload driver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F  # noqa: N812

from lcblm.eval.metrics import feature_recovery

if TYPE_CHECKING:
    from torch import Tensor, nn

# Models whose activation is a soft gate in [0, 1] (threshold at 0.5) vs. those
# whose activation is an unbounded non-negative latent (threshold just above 0).
_GATE_MODELS = ("vaee", "vaee_shared_encoder")
_SAE_MODELS = ("topk_sae", "sae_concept")
_SUPPORTED = (*_GATE_MODELS, *_SAE_MODELS)


def _default_active_thresh(model_name: str) -> float:
    if model_name in _GATE_MODELS:
        return 0.5
    return 1e-6


def _forward_parts(
    model: nn.Module, model_name: str, x: Tensor
) -> tuple[Tensor, Callable[[Tensor], Tensor], Tensor]:
    """Return ``(acts, decode_fn, recon)`` for a model.

    ``acts`` is the ``(B, K)`` per-concept activation strength, ``decode_fn`` maps
    a modified ``(B, K)`` activation tensor to a reconstruction, and ``recon`` is
    the reconstruction at the unmodified activations (the eval forward pass).
    """
    if model_name == "vaee":
        mu = model.encode(x)  # (B, K, E)
        alpha = torch.sigmoid(model._compute_logits(mu))  # (B, K)  # noqa: SLF001

        def decode_fn(a: Tensor) -> Tensor:
            return model.decode(a.unsqueeze(-1) * mu)

        return alpha, decode_fn, decode_fn(alpha)

    if model_name == "vaee_shared_encoder":
        e = model._encoder(x)  # (B, d)  # noqa: SLF001
        alpha = torch.sigmoid(model._compute_logits(e))  # (B, K)  # noqa: SLF001
        proto = model.prototypes.unsqueeze(0)  # (1, K, d)

        def decode_fn(a: Tensor) -> Tensor:
            return model.decode(a.unsqueeze(-1) * proto)

        return alpha, decode_fn, decode_fn(alpha)

    if model_name in _SAE_MODELS:
        _, z = model.encode(x)  # (B, L)

        def decode_fn(a: Tensor) -> Tensor:
            return model.decode(a)

        return z, decode_fn, decode_fn(z)

    msg = (
        f"intervention metrics not supported for model {model_name!r} "
        f"(supported: {_SUPPORTED})"
    )
    raise ValueError(msg)


@torch.no_grad()
def ablation_directions(
    model: nn.Module,
    model_name: str,
    x: Tensor,
    *,
    active_thresh: float | None = None,
    min_active: int = 8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-concept ablation statistics.

    For every concept ``k`` and every input, compute the ablation contribution
    ``delta_k = recon - decode(a with a_k := 0)`` and keep only the inputs where
    ``k`` is active (``a_k > active_thresh``).

    Returns:
        mean_dir: ``(K, D)`` mean of the *unit* ablation directions over active
            inputs (zero row for concepts active on fewer than ``min_active``
            inputs).
        resultant: ``(K,)`` resultant length of those unit directions in
            ``[0, 1]`` — the consistency of concept ``k`` (NaN if too few active).
        valid: ``(K,)`` bool mask of concepts with enough active inputs.
    """
    model.eval()
    acts, decode_fn, recon = _forward_parts(model, model_name, x)
    thresh = (
        _default_active_thresh(model_name) if active_thresh is None else active_thresh
    )
    k_dim = acts.shape[1]
    d_dim = recon.shape[1]
    dev = recon.device

    mean_dir = torch.zeros(k_dim, d_dim, device=dev)
    resultant = torch.full((k_dim,), float("nan"), device=dev)
    valid = torch.zeros(k_dim, dtype=torch.bool, device=dev)

    for k in range(k_dim):
        active = acts[:, k] > thresh
        n = int(active.sum().item())
        if n < min_active:
            continue
        a0 = acts.clone()
        a0[:, k] = 0.0
        delta = (recon - decode_fn(a0))[active]  # (n, D)
        unit = F.normalize(delta, dim=1)
        resultant_vec = unit.mean(dim=0)  # (D,)
        mean_dir[k] = F.normalize(resultant_vec, dim=0)
        resultant[k] = resultant_vec.norm()
        valid[k] = True

    return mean_dir, resultant, valid


@torch.no_grad()
def intervention_consistency(
    model: nn.Module,
    model_name: str,
    x: Tensor,
    *,
    active_thresh: float | None = None,
    min_active: int = 8,
) -> dict[str, float]:
    """Metric 6 — mean ablation-direction consistency over concepts.

    Returns ``mean_consistency`` (mean resultant length over valid concepts),
    ``min_consistency`` and ``n_concepts`` (how many concepts were evaluated).
    """
    _, resultant, valid = ablation_directions(
        model, model_name, x, active_thresh=active_thresh, min_active=min_active
    )
    vals = resultant[valid]
    if vals.numel() == 0:
        return {
            "mean_consistency": float("nan"),
            "min_consistency": float("nan"),
            "n_concepts": 0,
        }
    return {
        "mean_consistency": float(vals.mean().item()),
        "min_consistency": float(vals.min().item()),
        "n_concepts": int(valid.sum().item()),
    }


@torch.no_grad()
def intervention_factor_recovery(
    model: nn.Module,
    model_name: str,
    x: Tensor,
    atoms: Tensor,
    *,
    threshold: float = 0.7,
    active_thresh: float | None = None,
    min_active: int = 8,
) -> dict[str, float]:
    """Metric 7 — causal feature recovery from intervention directions.

    Hungarian-matches each concept's mean ablation direction to the ground-truth
    ``atoms`` (reusing :func:`feature_recovery`) and reports a per-concept
    dominance ratio (top atom cosine / second-best).
    """
    mean_dir, _, valid = ablation_directions(
        model, model_name, x, active_thresh=active_thresh, min_active=min_active
    )
    atoms = atoms.to(mean_dir.device).float()

    rec = feature_recovery(mean_dir, atoms, threshold=threshold)

    # Dominance ratio over the valid concepts only.
    dirs = mean_dir[valid]
    if dirs.shape[0] == 0:
        return {
            "causal_matched_fraction": rec["matched_fraction"],
            "causal_mean_cosine_sim": rec["mean_cosine_sim"],
            "mean_dominance": float("nan"),
            "n_concepts": 0,
        }
    cos = F.normalize(dirs, dim=1) @ F.normalize(atoms, dim=1).T  # (n_valid, n_atoms)
    top2 = cos.abs().topk(min(2, cos.shape[1]), dim=1).values
    if top2.shape[1] == 2:
        dominance = top2[:, 0] / top2[:, 1].clamp(min=1e-6)
    else:
        dominance = torch.full((top2.shape[0],), float("inf"))
    return {
        "causal_matched_fraction": rec["matched_fraction"],
        "causal_mean_cosine_sim": rec["mean_cosine_sim"],
        "mean_dominance": float(dominance.mean().item()),
        "n_concepts": int(valid.sum().item()),
    }


@torch.no_grad()
def evaluate_intervention(
    model: nn.Module,
    model_name: str,
    x: Tensor,
    atoms: Tensor,
    *,
    threshold: float = 0.7,
    active_thresh: float | None = None,
    min_active: int = 8,
) -> dict[str, float]:
    """Convenience: both intervention metrics in one dict."""
    cons = intervention_consistency(
        model, model_name, x, active_thresh=active_thresh, min_active=min_active
    )
    fac = intervention_factor_recovery(
        model,
        model_name,
        x,
        atoms,
        threshold=threshold,
        active_thresh=active_thresh,
        min_active=min_active,
    )
    return {**cons, **{k: v for k, v in fac.items() if k != "n_concepts"}}

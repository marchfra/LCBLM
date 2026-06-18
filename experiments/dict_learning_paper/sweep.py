"""Pareto sweep for the dict-learning paper.

Trains 5 models x N sparsity values on one dataset; results feed ``ct-plot``
to produce the L0-MSE Pareto figure.

The sweep axis is declared directly in the TOML as a **list-valued field**,
e.g. ``pi = [0.0625, 0.125, 0.25]`` for VAEE — no separate
``sparsity_param`` / ``sparsity_values`` keys needed.  The script detects
whichever field holds a list (there must be exactly one per model section).

Per-run output layout (one subdirectory per run, compatible with ``ct-plot``)::

    {output_dir}/{dataset}/{model}_{param}={val}/
        config.json
        results.json
        checkpoint.pt

Usage
-----
    dl-sweep --config experiments/dict_learning_paper/configs/sweep_synthetic.toml
    dl-sweep --config <path> --out-dir /fast/outputs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tomllib
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import wandb
from lcblm.data.image_loaders import load_dsprites, load_fmnist, load_mnist
from lcblm.data.synthetic import make_complex_synthetic, make_synthetic
from lcblm.eval.metrics import feature_recovery
from lcblm.training.configs import MODEL_CONFIG_CLASSES, VAEEConfig, _BaseConfig
from lcblm.training.loops import (
    train_beta_vae,
    train_sae_concept,
    train_topk_sae,
    train_vaee,
    train_vaee_shared_encoder,
    train_vq_vae,
)
from lcblm.utils import get_device, set_seeds
from lcblm.utils.data import FlatTensorDataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from lcblm.training.loops import RunResult

# ── Training dispatch ─────────────────────────────────────────────────────────

_TRAIN_FNS: dict[str, Callable[..., tuple[nn.Module, RunResult]]] = {
    "vaee": train_vaee,
    "vaee_shared_encoder": train_vaee_shared_encoder,
    "topk_sae": train_topk_sae,
    "sae_concept": train_sae_concept,
    "vq_vae": train_vq_vae,
    "beta_vae": train_beta_vae,
}

# Canonical order; models absent from the TOML are silently skipped.
_MODEL_ORDER = (
    "vaee",
    "vaee_shared_encoder",
    "topk_sae",
    "sae_concept",
    "vq_vae",
    "beta_vae",
)

# _BaseConfig fields that may appear in the TOML; 'device' is excluded because
# it is resolved automatically via the field's default_factory (get_device).
_BASE_FIELDS: frozenset[str] = frozenset(_BaseConfig.__dataclass_fields__) - {"device"}

# ── Dataset loading ───────────────────────────────────────────────────────────

_FLAT_IMAGE_LOADERS: dict[str, Callable[..., FlatTensorDataset]] = {
    "mnist": load_mnist,
    "fmnist": load_fmnist,
}


def _load_dataset(
    dataset: str,
    data_path: str | None,
    synthetic_cfg: dict | None = None,
) -> tuple[FlatTensorDataset, FlatTensorDataset, torch.Tensor | None]:
    """Return (train_ds, val_ds, gt_features).

    gt_features is the [n_features, input_dim] ground-truth atom matrix for
    synthetic datasets; None for all other datasets.
    """
    if dataset in ("synthetic", "complex_synthetic"):
        gen = make_synthetic if dataset == "synthetic" else make_complex_synthetic
        full, features = gen(**(synthetic_cfg or {}))
        n = len(full)
        cut = int(0.8 * n)
        return (
            FlatTensorDataset(full.data[:cut]),
            FlatTensorDataset(full.data[cut:]),
            features,
        )

    if data_path is None:
        msg = "data_path is required for all datasets other than 'synthetic'"
        raise ValueError(msg)

    if dataset in _FLAT_IMAGE_LOADERS:
        load_fn = _FLAT_IMAGE_LOADERS[dataset]
        return load_fn(data_path, "train"), load_fn(data_path, "val"), None

    if dataset == "dsprites":
        train_ds, _ = load_dsprites(data_path, "train")
        val_ds, _ = load_dsprites(data_path, "val")
        return train_ds, val_ds, None

    msg = f"Unknown dataset: {dataset!r}"
    raise ValueError(msg)


# ── Prototype / concept extraction (synthetic 2D only) ────────────────────────


def _extract_prototypes(model_name: str, model: nn.Module) -> list[list[float]] | None:
    """Return learned prototypes in input space, one row per concept."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        if model_name in ("topk_sae", "sae_concept"):
            return model._decoder.weight.T.cpu().tolist()
        if model_name == "vq_vae":
            return model._decoder(model.codebook).cpu().tolist()
        if model_name == "beta_vae":
            return model._decoder.weight.T.cpu().tolist()
        if model_name in ("vaee", "vaee_shared_encoder"):
            K = model.num_embeddings
            E = model.embedding_size
            z_in = torch.zeros(K, K, E, device=dev)
            idx = torch.arange(K, device=dev)
            z_in[idx, idx] = model.prototypes
            return model._decoder(z_in.flatten(start_dim=1)).cpu().tolist()
    return None


def _dominant_concept(
    model_name: str, model: nn.Module, val_ds: FlatTensorDataset
) -> list[int]:
    """Return per-sample dominant concept index for scatter plot colouring."""
    model.eval()
    dev = next(model.parameters()).device
    data = val_ds.data.to(dev)
    with torch.no_grad():
        if model_name in ("topk_sae", "sae_concept"):
            return model(data).latents.argmax(dim=1).cpu().tolist()
        if model_name == "vq_vae":
            return model(data).indices.cpu().tolist()
        if model_name == "beta_vae":
            return model(data).mu.abs().argmax(dim=1).cpu().tolist()
        if model_name in ("vaee", "vaee_shared_encoder"):
            return model(data).alpha.argmax(dim=1).cpu().tolist()
    return []


def _concept_weights(
    model_name: str, model: nn.Module, val_ds: FlatTensorDataset
) -> list[list[float]]:
    """Per-sample, per-concept activation weights (``alpha``) for colour blending.

    Returns an ``[N, K]`` matrix of non-negative weights so the 2D plot can colour
    each point as a weighted sum of its active concepts' colours. Uses the same
    per-model activation that drives ``_dominant_concept`` (VAEE gates, SAE latents,
    one-hot codes for VQ-VAE, ``|mu|`` for β-VAE).
    """
    model.eval()
    dev = next(model.parameters()).device
    data = val_ds.data.to(dev)
    with torch.no_grad():
        if model_name in ("topk_sae", "sae_concept"):
            w = model(data).latents.clamp(min=0.0)
        elif model_name == "vq_vae":
            idx = model(data).indices
            k = model.codebook.shape[0]
            w = torch.zeros(idx.shape[0], k, device=dev)
            w[torch.arange(idx.shape[0], device=dev), idx] = 1.0
        elif model_name == "beta_vae":
            w = model(data).mu.abs()
        elif model_name in ("vaee", "vaee_shared_encoder"):
            w = model(data).alpha
        else:
            return []
    return w.cpu().tolist()


def _alive_concepts(
    model_name: str, model: nn.Module, val_ds: FlatTensorDataset, threshold: float
) -> list[int]:
    """Concept indices that fire on >= 0.1% of val samples.

    Mirrors the per-model activation extraction used by ``alive_dict_size`` so the
    2D plot draws an arrow for every concept the metric counts as alive — not only
    those that win the per-sample argmax (which undercounts VAEE's independent gates).
    """
    model.eval()
    dev = next(model.parameters()).device
    data = val_ds.data.to(dev)
    with torch.no_grad():
        if model_name in ("topk_sae", "sae_concept"):
            fired = model(data).latents > 0
        elif model_name == "vq_vae":
            idx = model(data).indices
            k = model.codebook.shape[0]
            fired = torch.zeros(idx.shape[0], k, dtype=torch.bool, device=dev)
            fired[torch.arange(idx.shape[0], device=dev), idx] = True
        elif model_name == "beta_vae":
            fired = model(data).mu.abs() > threshold
        elif model_name in ("vaee", "vaee_shared_encoder"):
            fired = model(data).c > threshold
        else:
            return []
    n = fired.shape[0]
    fire_counts = fired.float().sum(0)  # [K]
    alive = (fire_counts >= 0.001 * n).nonzero(as_tuple=True)[0]
    return alive.cpu().tolist()


# ── Config building ───────────────────────────────────────────────────────────


def _find_sweep(cfg_cls: type, model_raw: dict) -> tuple[str, list]:
    """Return *(param_name, values)* for the single list-valued sweep axis.

    Scans *model_raw* for any field that is both a known field of *cfg_cls*
    (or *_BaseConfig*) and holds a list.  Raises ``ValueError`` if there is
    not exactly one such field.
    """
    known = _BASE_FIELDS | frozenset(cfg_cls.__dataclass_fields__)  # ty:ignore[unresolved-attribute]
    sweeps = {k: v for k, v in model_raw.items() if k in known and isinstance(v, list)}
    if len(sweeps) != 1:
        found = list(sweeps) if sweeps else "none"
        msg = (
            f"[{cfg_cls.__name__}] must contain exactly one list-valued field "
            f"(the sweep axis); found: {found}"
        )
        raise ValueError(msg)
    param, values = next(iter(sweeps.items()))
    return param, values


def _make_cfg(
    cfg_cls: type,
    global_raw: dict,
    model_raw: dict,
    sweep_param: str,
    sweep_val: object,
) -> _BaseConfig:
    """Build one config object for a single (model, sparsity_val) run.

    Merge priority (later wins):
        1. global TOML base fields
        2. model-section scalar fields
        3. the single swept value

    Device is not passed explicitly — the config's ``default_factory``
    (``get_device``) resolves it at instantiation time.
    """
    known = _BASE_FIELDS | frozenset(cfg_cls.__dataclass_fields__)  # ty:ignore[unresolved-attribute]
    raw: dict[str, Any] = {
        **{k: v for k, v in global_raw.items() if k in known},
        **{
            k: v for k, v in model_raw.items() if k in known and not isinstance(v, list)
        },
        sweep_param: sweep_val,
    }
    if cfg_cls is VAEEConfig:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cfg_cls(**raw)
    return cfg_cls(**raw)


# ── Output ────────────────────────────────────────────────────────────────────


def _save_run(
    model_name: str,
    model: nn.Module,
    cfg: _BaseConfig,
    result: RunResult,
    out_dir: Path,
    extra: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict.pop("device", None)
    with (out_dir / "config.json").open("w") as f:
        json.dump({"model_type": model_name, "run_config": cfg_dict}, f, indent=2)

    result_dict = dataclasses.asdict(result)
    if extra:
        result_dict.update(extra)
    with (out_dir / "results.json").open("w") as f:
        json.dump(result_dict, f, indent=2)

    torch.save(model.state_dict(), out_dir / "checkpoint.pt")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dl-sweep",
        description="Pareto sweep for the dict-learning paper.",
    )
    parser.add_argument("--config", required=True, help="Path to sweep TOML config.")
    parser.add_argument(
        "--out-dir",
        "-o",
        default=None,
        help="Override output root directory (default: value from TOML or outputs/).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the top-level batch_size and tag the output dir (for ablations).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=f"Restrict the sweep to these model sections (choices: {', '.join(_MODEL_ORDER)}).",
    )
    args = parser.parse_args()

    if args.models is not None:
        unknown = [m for m in args.models if m not in _MODEL_ORDER]
        if unknown:
            msg = f"Unknown model(s): {unknown}. Choose from {list(_MODEL_ORDER)}."
            raise SystemExit(msg)

    with Path(args.config).open("rb") as f:
        raw = tomllib.load(f)

    if args.batch_size is not None:
        raw["batch_size"] = args.batch_size

    dataset: str = raw["dataset"]
    data_path: str | None = raw.get("data_path")
    default_out = raw.get("output_dir", "experiments/dict_learning_paper/outputs")
    config_stem = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_dir or default_out) / config_stem / timestamp

    print(f"Dataset : {dataset}")
    print(f"Device  : {get_device()}")
    print(f"Output  : {out_root}\n")

    train_ds, val_ds, gt_features = _load_dataset(
        dataset, data_path, raw.get("synthetic")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if gt_features is not None:
        with (out_root / "ground_truth.json").open("w") as f:
            json.dump(
                {
                    "atoms_2d": gt_features.cpu().tolist(),
                    "val_data_2d": val_ds.data.cpu().tolist(),
                },
                f,
                indent=2,
            )

    for model_name in _MODEL_ORDER:
        if args.models is not None and model_name not in args.models:
            continue
        model_raw: dict | None = raw.get(model_name)
        if model_raw is None:
            continue

        cfg_cls = MODEL_CONFIG_CLASSES[model_name]
        sweep_param, sweep_values = _find_sweep(cfg_cls, model_raw)
        n_runs = len(sweep_values)

        for i, sweep_val in enumerate(sweep_values, 1):
            run_label = f"{sweep_param}={sweep_val}"
            print(f"[{model_name}  {i}/{n_runs}  {run_label}]")

            cfg = _make_cfg(cfg_cls, raw, model_raw, sweep_param, sweep_val)
            set_seeds(cfg.seed)

            wandb_run: wandb.sdk.wandb_run.Run | None = None
            if cfg.wandb_project:
                wandb_run = wandb.init(
                    project=cfg.wandb_project,
                    group=dataset,
                    name=f"{model_name}_{run_label}",
                    tags=[dataset, model_name],
                    config=dataclasses.asdict(cfg)
                    | {"model_type": model_name, "dataset": dataset},
                    reinit=True,
                )

            model, result = _TRAIN_FNS[model_name](train_ds, val_ds, cfg, wandb_run)

            extra: dict | None = None
            if gt_features is not None:
                atoms_2d = _extract_prototypes(model_name, model)
                if atoms_2d is not None:
                    rec = feature_recovery(
                        torch.tensor(atoms_2d, dtype=torch.float32),
                        gt_features.cpu(),
                        threshold=float(raw.get("recovery_threshold", 0.9)),
                    )
                    result.matched_fraction = rec["matched_fraction"]
                    result.mean_cosine_sim = rec["mean_cosine_sim"]
                extra = {
                    "prototypes_2d": atoms_2d,
                    "val_dominant_concept": _dominant_concept(
                        model_name, model, val_ds
                    ),
                    "val_concept_weights": _concept_weights(model_name, model, val_ds),
                    "alive_concepts": _alive_concepts(
                        model_name, model, val_ds, getattr(cfg, "l0_threshold", 0.0)
                    ),
                }

            out_dir = out_root / f"{model_name}_{run_label}"
            _save_run(model_name, model, cfg, result, out_dir, extra)

            if wandb_run is not None:
                artifact = wandb.Artifact(
                    name=f"{model_name}-{run_label.replace('=', '_')}",
                    type="run_result",
                )
                artifact.add_file(str(out_dir / "results.json"))
                wandb_run.log_artifact(artifact)
                wandb_run.finish()

            rec_str = (
                f"  matched={result.matched_fraction:.2f}  cos={result.mean_cosine_sim:.3f}"
                if result.matched_fraction is not None
                else ""
            )
            print(
                f"  alive={result.alive_dict_size}"
                f"  l0={result.best_l0:.2f}"
                f"  mse={result.best_val_recon:.4f}" + rec_str + f"  -> {out_dir.name}",
            )


if __name__ == "__main__":
    main()

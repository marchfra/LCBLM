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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

import wandb
from lcblm.data.image_loaders import load_dsprites, load_fmnist, load_mnist
from lcblm.data.synthetic import make_synthetic
from lcblm.training.configs import MODEL_CONFIG_CLASSES, VAEEConfig, _BaseConfig
from lcblm.training.loops import (
    train_beta_vae,
    train_sae_concept,
    train_topk_sae,
    train_vaee,
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
    "topk_sae": train_topk_sae,
    "sae_concept": train_sae_concept,
    "vq_vae": train_vq_vae,
    "beta_vae": train_beta_vae,
}

# Canonical order; models absent from the TOML are silently skipped.
_MODEL_ORDER = ("vaee", "topk_sae", "sae_concept", "vq_vae", "beta_vae")

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
) -> tuple[FlatTensorDataset, FlatTensorDataset]:
    if dataset == "synthetic":
        full, _ = make_synthetic(**(synthetic_cfg or {}))
        n = len(full)
        cut = int(0.8 * n)
        return FlatTensorDataset(full.data[:cut]), FlatTensorDataset(full.data[cut:])

    if data_path is None:
        msg = "data_path is required for all datasets other than 'synthetic'"
        raise ValueError(msg)

    if dataset in _FLAT_IMAGE_LOADERS:
        load_fn = _FLAT_IMAGE_LOADERS[dataset]
        return load_fn(data_path, "train"), load_fn(data_path, "val")

    if dataset == "dsprites":
        train_ds, _ = load_dsprites(data_path, "train")
        val_ds, _ = load_dsprites(data_path, "val")
        return train_ds, val_ds

    msg = f"Unknown dataset: {dataset!r}"
    raise ValueError(msg)


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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict.pop("device", None)
    with (out_dir / "config.json").open("w") as f:
        json.dump({"model_type": model_name, "run_config": cfg_dict}, f, indent=2)

    with (out_dir / "results.json").open("w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2)

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
    args = parser.parse_args()

    with Path(args.config).open("rb") as f:
        raw = tomllib.load(f)

    dataset: str = raw["dataset"]
    data_path: str | None = raw.get("data_path")
    default_out = raw.get("output_dir", "experiments/dict_learning_paper/outputs")
    out_root = Path(args.out_dir or default_out) / dataset

    print(f"Dataset : {dataset}")
    print(f"Device  : {get_device()}")
    print(f"Output  : {out_root}\n")

    train_ds, val_ds = _load_dataset(dataset, data_path, raw.get("synthetic"))
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name in _MODEL_ORDER:
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

            out_dir = out_root / f"{model_name}_{run_label}"
            _save_run(model_name, model, cfg, result, out_dir)

            if wandb_run is not None:
                artifact = wandb.Artifact(
                    name=f"{model_name}_{run_label}",
                    type="run_result",
                )
                artifact.add_file(str(out_dir / "results.json"))
                wandb_run.log_artifact(artifact)
                wandb_run.finish()

            print(
                f"  alive={result.alive_dict_size}"
                f"  l0={result.best_l0:.2f}"
                f"  mse={result.best_val_recon:.4f}"
                f"  -> {out_dir.name}",
            )


if __name__ == "__main__":
    main()

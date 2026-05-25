"""Pareto sweep: 5 models × 3 sparsity points = 15 runs per dataset.

Usage:
    python experiments/dict_learning_paper/sweep.py \\
        --config experiments/dict_learning_paper/configs/sweep_synthetic.toml \\
        --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import torch

from lcblm.data.synthetic import make_synthetic
from lcblm.training.configs import (
    BetaVAEConfig,
    SAEConceptConfig,
    TopKSAEConfig,
    VAEEConfig,
    VQVAEConfig,
)
from lcblm.training.loops import (
    RunResult,
    train_beta_vae,
    train_sae_concept,
    train_topk_sae,
    train_vaee,
    train_vq_vae,
)
from lcblm.utils.data import FlatTensorDataset

# ── Dataset loaders ───────────────────────────────────────────────────────────


def _load_datasets(
    dataset: str,
    data_path: str | None,
    device: torch.device,
    synthetic_cfg: dict | None = None,
) -> tuple[FlatTensorDataset, FlatTensorDataset]:
    if dataset == "synthetic":
        full, _ = make_synthetic(**(synthetic_cfg or {}))
        n = len(full)
        cut = int(0.8 * n)
        return FlatTensorDataset(full.data[:cut]), FlatTensorDataset(full.data[cut:])
    from lcblm.data.image_loaders import load_dsprites, load_fmnist, load_mnist
    if dataset == "mnist":
        assert data_path, "--data-path required for mnist"
        return load_mnist(data_path, "train"), load_mnist(data_path, "val")
    if dataset == "fmnist":
        assert data_path, "--data-path required for fmnist"
        return load_fmnist(data_path, "train"), load_fmnist(data_path, "val")
    if dataset == "dsprites":
        assert data_path, "--data-path required for dsprites"
        train_ds, _ = load_dsprites(data_path, "train")
        val_ds, _ = load_dsprites(data_path, "val")
        return train_ds, val_ds
    msg = f"Unknown dataset: {dataset!r}"
    raise ValueError(msg)


# ── Per-model run ─────────────────────────────────────────────────────────────


_SHARED_KEYS = frozenset({"epochs", "lr", "batch_size", "early_stopping_patience"})


def _run_one(
    model_name: str,
    sparsity_key: str,
    sparsity_val: float | int,
    model_cfg: dict,
    fixed: dict,
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    device: torch.device,
) -> RunResult:
    overrides = {k: v for k, v in model_cfg.items() if k in _SHARED_KEYS}
    common = {**fixed, **overrides, "device": device}

    if model_name == "vaee":
        cfg = VAEEConfig(
            **common,
            num_embeddings=model_cfg["num_embeddings"],
            embedding_size=model_cfg["embedding_size"],
            pi=float(sparsity_val),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, result = train_vaee(train_ds, val_ds, cfg)

    elif model_name == "topk_sae":
        topk_extras = {
            key: model_cfg[key]
            for key in ("latent_dim", "k_aux")
            if key in model_cfg
        }
        cfg = TopKSAEConfig(**common, k=int(sparsity_val), **topk_extras)
        _, result = train_topk_sae(train_ds, val_ds, cfg)

    elif model_name == "sae_concept":
        cfg = SAEConceptConfig(
            **common,
            vaee_num_embeddings=model_cfg["vaee_num_embeddings"],
            lambda_l1=float(sparsity_val),
        )
        _, result = train_sae_concept(train_ds, val_ds, cfg)

    elif model_name == "vq_vae":
        vq_extras = {
            key: model_cfg[key]
            for key in ("embedding_dim",)
            if key in model_cfg
        }
        cfg = VQVAEConfig(**common, num_codes=int(sparsity_val), **vq_extras)
        _, result = train_vq_vae(train_ds, val_ds, cfg)

    elif model_name == "beta_vae":
        cfg = BetaVAEConfig(
            **common,
            latent_dim=model_cfg["latent_dim"],
            beta=float(sparsity_val),
            kl_warmup_epochs=int(model_cfg.get("kl_warmup_epochs", 0)),
        )
        _, result = train_beta_vae(train_ds, val_ds, cfg)

    else:
        msg = f"Unknown model: {model_name!r}"
        raise ValueError(msg)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

_MODEL_NAMES = ("vaee", "topk_sae", "sae_concept", "vq_vae", "beta_vae")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dict-learning Pareto sweep.")
    parser.add_argument("--config", required=True, help="Path to sweep TOML config.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = tomllib.load(fh)

    dataset: str = cfg["dataset"]
    output_dir: str = cfg.get("output_dir", "experiments/dict_learning_paper/outputs")
    data_path: str | None = cfg.get("data_path")

    fixed = {
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "early_stopping_patience": cfg["early_stopping_patience"],
    }

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    synthetic_cfg = cfg.get("synthetic")
    train_ds, val_ds = _load_datasets(dataset, data_path, device, synthetic_cfg)

    out_root = Path(output_dir) / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name in _MODEL_NAMES:
        model_cfg: dict = cfg.get(model_name, {})
        sparsity_param: str = model_cfg["sparsity_param"]
        sparsity_values: list = model_cfg["sparsity_values"]

        for sparsity_val in sparsity_values:
            result = _run_one(
                model_name, sparsity_param, sparsity_val,
                model_cfg, fixed, train_ds, val_ds, device,
            )

            out_path = out_root / f"{model_name}_{sparsity_val}.json"
            with out_path.open("w") as fh:
                json.dump(dataclasses.asdict(result), fh, indent=2)

            print(
                f"[{model_name}/{sparsity_param}={sparsity_val}]"
                f" alive={result.alive_dict_size}"
                f" l0={result.best_l0:.2f}"
                f" mse={result.best_val_recon:.4f}"
            )


if __name__ == "__main__":
    main()

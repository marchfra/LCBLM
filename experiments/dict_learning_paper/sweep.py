"""Pareto sweep: 5 models × 3 sparsity points = 15 runs per dataset.

Usage:
    python experiments/dict_learning_paper/sweep.py \\
        --dataset synthetic \\
        --output-dir experiments/dict_learning_paper/outputs \\
        --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import warnings
from pathlib import Path

import torch

from lcblm.data.image_loaders import load_dsprites, load_fmnist, load_mnist
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

# ── Sparsity grid ─────────────────────────────────────────────────────────────

_GRID: dict[str, list[tuple[str, float | int]]] = {
    "vaee":        [("pi", v) for v in [0.02, 0.05, 0.10]],
    "topk_sae":    [("k", v) for v in [4, 8, 16]],
    "sae_concept": [("lambda_l1", v) for v in [0.01, 0.05, 0.10]],
    "vq_vae":      [("num_codes", v) for v in [64, 128, 256]],
    "beta_vae":    [("beta", v) for v in [1.0, 4.0, 8.0]],
}

_FIXED = dict(epochs=50, lr=1e-3, batch_size=512, early_stopping_patience=5)

# ── Dataset loaders ───────────────────────────────────────────────────────────


def _load_datasets(
    dataset: str,
    data_path: str | None,
    device: torch.device,
) -> tuple[FlatTensorDataset, FlatTensorDataset]:
    if dataset == "synthetic":
        full, _ = make_synthetic()
        n = len(full)
        cut = int(0.8 * n)
        return FlatTensorDataset(full.data[:cut]), FlatTensorDataset(full.data[cut:])
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


def _run_one(
    model_name: str,
    sparsity_key: str,
    sparsity_val: float | int,
    train_ds: FlatTensorDataset,
    val_ds: FlatTensorDataset,
    device: torch.device,
) -> RunResult:
    common = dict(**_FIXED, device=device)

    if model_name == "vaee":
        cfg = VAEEConfig(
            **common,
            num_embeddings=64,
            embedding_size=16,
            pi=float(sparsity_val),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, result = train_vaee(train_ds, val_ds, cfg)

    elif model_name == "topk_sae":
        cfg = TopKSAEConfig(**common, k=int(sparsity_val))
        _, result = train_topk_sae(train_ds, val_ds, cfg)

    elif model_name == "sae_concept":
        cfg = SAEConceptConfig(
            **common,
            vaee_num_embeddings=64,
            lambda_l1=float(sparsity_val),
        )
        _, result = train_sae_concept(train_ds, val_ds, cfg)

    elif model_name == "vq_vae":
        cfg = VQVAEConfig(**common, num_codes=int(sparsity_val))
        _, result = train_vq_vae(train_ds, val_ds, cfg)

    elif model_name == "beta_vae":
        cfg = BetaVAEConfig(**common, latent_dim=64, beta=float(sparsity_val))
        _, result = train_beta_vae(train_ds, val_ds, cfg)

    else:
        msg = f"Unknown model: {model_name!r}"
        raise ValueError(msg)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Dict-learning Pareto sweep.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["synthetic", "mnist", "fmnist", "dsprites"],
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--output-dir",
        default="experiments/dict_learning_paper/outputs",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    train_ds, val_ds = _load_datasets(args.dataset, args.data_path, device)

    out_root = Path(args.output_dir) / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    for model_name, sparsity_points in _GRID.items():
        for sparsity_key, sparsity_val in sparsity_points:
            result = _run_one(
                model_name, sparsity_key, sparsity_val, train_ds, val_ds, device
            )

            out_path = out_root / f"{model_name}_{sparsity_val}.json"
            with out_path.open("w") as fh:
                json.dump(dataclasses.asdict(result), fh, indent=2)

            print(
                f"[{model_name}/{sparsity_key}={sparsity_val}]"
                f" alive={result.alive_dict_size}"
                f" l0={result.best_l0:.2f}"
                f" mse={result.best_val_recon:.4f}"
            )


if __name__ == "__main__":
    main()

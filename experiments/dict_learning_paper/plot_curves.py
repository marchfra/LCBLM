"""Learning-curve plot for dict-learning Pareto sweep outputs.

Usage:
    python experiments/dict_learning_paper/plot_curves.py \
        experiments/dict_learning_paper/outputs/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_MODEL_DISPLAY = {
    "vaee": "VAEE",
    "vaee_shared_encoder": "VAEE-Shared Encoder",
    "topk_sae": "TopK-SAE",
    "sae_concept": "L1-SAE",
    "vq_vae": "VQ-VAE",
    "beta_vae": "β-VAE",
}
_ORDER = list(_MODEL_DISPLAY)
_CMAP = mpl.colormaps["viridis"]


def _load(out_dir: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {m: [] for m in _ORDER}
    for path in sorted(out_dir.glob("*/results.json")):
        data = json.loads(path.read_text())
        model = data["model_name"]
        if model in runs:
            data["_label"] = path.parent.name[len(model) + 1 :]
            runs[model].append(data)
    return runs


def plot(out_dir: Path) -> None:
    runs = _load(out_dir)
    n_models = len(_ORDER)
    # Share x (epochs) globally; share y per metric group (cols 0,2 = recon MSE;
    # cols 1,3 = L0) so the same metric is on one scale across all models.
    fig, axes = plt.subplots(
        n_models,
        4,
        figsize=(16, 3 * n_models),
        squeeze=False,
        sharex=True,
    )
    fig.suptitle(f"Learning curves — {out_dir.name}", fontsize=14)

    for row, model_key in enumerate(_ORDER):
        ax_train_mse, ax_train_l0, ax_val_mse, ax_val_l0 = axes[row]
        model_runs = runs[model_key]
        colors = _CMAP(np.linspace(0.2, 0.9, max(len(model_runs), 1)))

        for run, color in zip(model_runs, colors, strict=False):
            n_epochs = len(run["val_recon"])
            epochs = range(1, n_epochs + 1)
            label = run["_label"]

            best_epoch = int(np.argmin(run["val_recon"])) + 1

            ax_train_mse.plot(
                epochs, run["train_recon"], color=color, label=label, linewidth=1.5
            )
            ax_train_mse.axvline(best_epoch, color=color, linewidth=0.8, linestyle=":")

            if run.get("train_l0") and run["train_l0"][0] > 0:
                ax_train_l0.plot(
                    epochs, run["train_l0"], color=color, label=label, linewidth=1.5
                )
                ax_train_l0.axvline(
                    best_epoch, color=color, linewidth=0.8, linestyle=":"
                )

            ax_val_mse.plot(
                epochs, run["val_recon"], color=color, label=label, linewidth=1.5
            )
            ax_val_mse.axvline(best_epoch, color=color, linewidth=0.8, linestyle=":")

            if run.get("val_l0") and run["val_l0"][0] > 0:
                ax_val_l0.plot(
                    epochs, run["val_l0"], color=color, label=label, linewidth=1.5
                )
                ax_val_l0.axvline(best_epoch, color=color, linewidth=0.8, linestyle=":")

        ax_train_mse.set_title(_MODEL_DISPLAY[model_key], fontsize=11, loc="left")
        ax_train_mse.set_ylabel("Train recon MSE")
        ax_train_l0.set_ylabel("Train # active concepts")
        ax_val_mse.set_ylabel("Val recon MSE")
        ax_val_l0.set_ylabel("Val # active concepts")
        if row == n_models - 1:
            ax_train_mse.set_xlabel("Epoch")
            ax_train_l0.set_xlabel("Epoch")
            ax_val_mse.set_xlabel("Epoch")
            ax_val_l0.set_xlabel("Epoch")
        if model_runs:
            ax_train_mse.legend(fontsize=8, title="sparsity", framealpha=0.7)
            ax_val_mse.legend(fontsize=8, title="sparsity", framealpha=0.7)
            if run.get("train_l0") and run["train_l0"][0] > 0:
                ax_train_l0.legend(fontsize=8, title="sparsity", framealpha=0.7)
            if run.get("val_l0") and run["val_l0"][0] > 0:
                ax_val_l0.legend(fontsize=8, title="sparsity", framealpha=0.7)

    # Common y-scale per metric group so models are directly comparable:
    # recon-MSE columns (0, 2) share one scale; L0 columns (1, 3) share another.
    def _group_vals(keys: tuple[str, ...]) -> list[float]:
        return [
            v
            for model_runs in runs.values()
            for run in model_runs
            for key in keys
            for v in (run.get(key) or [])
            if v is not None
        ]

    mse_vals = _group_vals(("train_recon", "val_recon"))
    l0_max = max(_group_vals(("train_l0", "val_l0")) or [1.0])
    # MSE on a log scale so near-zero models (TopK/L1/VQ) are distinguishable
    # from the higher-MSE ones (VAEE-SE, β-VAE) across orders of magnitude.
    mse_pos = [v for v in mse_vals if v > 0] or [1e-4]
    mse_lo, mse_hi = min(mse_pos), max(mse_vals or [1.0])
    for row in range(n_models):
        for col in (0, 2):
            axes[row][col].set_yscale("log")
            axes[row][col].set_ylim(mse_lo * 0.8, mse_hi * 1.3)
        for col in (1, 3):
            axes[row][col].set_ylim(0, l0_max * 1.05)

    fig.tight_layout()
    out_path = out_dir / "learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_dir",
        help="Path to a dataset output dir, e.g. outputs/synthetic",
    )
    args = parser.parse_args()
    plot(Path(args.out_dir))


if __name__ == "__main__":
    main()

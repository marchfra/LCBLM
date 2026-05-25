"""Learning-curve plot for dict-learning Pareto sweep outputs.

Usage:
    python experiments/dict_learning_paper/plot_curves.py \
        experiments/dict_learning_paper/outputs/synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_MODEL_DISPLAY = {
    "vaee":        "VAEE",
    "topk_sae":    "TopK-SAE",
    "sae_concept": "L1-SAE",
    "vq_vae":      "VQ-VAE",
    "beta_vae":    "β-VAE",
}
_ORDER = list(_MODEL_DISPLAY)
_CMAP = plt.cm.viridis


def _load(out_dir: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {m: [] for m in _ORDER}
    for path in sorted(out_dir.glob("*.json")):
        data = json.loads(path.read_text())
        model = data["model_name"]
        if model in runs:
            data["_label"] = path.stem[len(model) + 1:]
            runs[model].append(data)
    return runs


def plot(out_dir: Path) -> None:
    runs = _load(out_dir)
    n_models = len(_ORDER)
    fig, axes = plt.subplots(
        n_models, 3,
        figsize=(14, 3 * n_models),
        squeeze=False,
        sharex=False,
    )
    fig.suptitle(f"Learning curves — {out_dir.name}", fontsize=14)

    for row, model_key in enumerate(_ORDER):
        ax_train, ax_val, ax_l0 = axes[row]
        model_runs = runs[model_key]
        colors = _CMAP(np.linspace(0.2, 0.9, max(len(model_runs), 1)))

        for run, color in zip(model_runs, colors):
            n_epochs = len(run["val_recon"])
            epochs = range(1, n_epochs + 1)
            label = run["_label"]

            best_epoch = int(np.argmin(run["val_recon"])) + 1

            ax_train.plot(epochs, run["train_recon"], color=color, label=label, linewidth=1.5)
            ax_train.axvline(best_epoch, color=color, linewidth=0.8, linestyle=":")

            ax_val.plot(epochs, run["val_recon"], color=color, label=label, linewidth=1.5)
            ax_val.axvline(best_epoch, color=color, linewidth=0.8, linestyle=":")

            if run["val_l0"] and run["val_l0"][0] > 0:
                ax_l0.plot(epochs, run["val_l0"], color=color, label=label, linewidth=1.5)
                ax_l0.axvline(best_epoch, color=color, linewidth=0.8, linestyle=":")

        ax_train.set_title(_MODEL_DISPLAY[model_key], fontsize=11, loc="left")
        ax_train.set_ylabel("Train recon MSE")
        ax_val.set_ylabel("Val recon MSE")
        ax_l0.set_ylabel("Val L0")
        if row == n_models - 1:
            ax_train.set_xlabel("Epoch")
            ax_val.set_xlabel("Epoch")
            ax_l0.set_xlabel("Epoch")
        if model_runs:
            ax_train.legend(fontsize=8, title="sparsity", framealpha=0.7)
            ax_val.legend(fontsize=8, title="sparsity", framealpha=0.7)
            if run["val_l0"] and run["val_l0"][0] > 0:
                ax_l0.legend(fontsize=8, title="sparsity", framealpha=0.7)

    fig.tight_layout()
    out_path = out_dir / "learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", help="Path to a dataset output dir, e.g. outputs/synthetic")
    args = parser.parse_args()
    plot(Path(args.out_dir))


if __name__ == "__main__":
    main()

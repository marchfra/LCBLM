"""Plotting utilities for the VAEE vs SparseAE experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    from .exp_config import DatasetConfig, RunConfig
    from .exp_training import RunResult


COLORS = {
    "VAEE": "coral",
    "SparseAE-concept": "steelblue",
    "SparseAE-param": "seagreen",
}
MARKERS = {
    "VAEE": "^",
    "SparseAE-concept": "o",
    "SparseAE-param": "s",
}
MODEL_NAMES = ("VAEE", "SparseAE-concept", "SparseAE-param")


def _scatter_panel(
    ax: plt.Axes,
    results: list[RunResult],
    model_names: tuple[str, ...],
) -> None:
    """Draw scatter points and n_concepts annotations for the given model names."""
    for model_name in model_names:
        rs = [r for r in results if r.model_name == model_name]
        if not rs:
            continue
        l0s = [r.best_l0 for r in rs]
        recons = [r.best_val_recon for r in rs]
        ax.scatter(
            l0s,
            recons,
            c=COLORS[model_name],
            marker=MARKERS[model_name],
            label=model_name,
        )
        for r, l0, rec in zip(rs, l0s, recons, strict=True):
            ax.annotate(
                f"${r.n_concepts}$",
                (l0, rec),
                textcoords="offset points",
                xytext=(4, 2),
                fontsize=11,
                color=COLORS[model_name],
            )


def plot_l0_recon(
    results: list[RunResult],
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
    out_dir: Path,
) -> None:
    """Two-panel scatter plot of L0 vs reconstruction MSE.

    Left panel: VAEE vs SparseAE-concept (concept-count-matched).
    Right panel: VAEE vs SparseAE-param (parameter-matched).
    VAEE points appear in both panels as the reference series.

    Args:
        results: Collected RunResult objects from run_experiment().
        cfg: Run hyperparameters (used for the figure title).
        ds_cfg: Dataset metadata (used for the figure title).
        out_dir: Directory to save the figure into.

    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    _scatter_panel(ax_left, results, ("VAEE", "SparseAE-concept"))
    _scatter_panel(ax_right, results, ("VAEE", "SparseAE-param"))

    for ax, subtitle in (
        (ax_left, "concept-matched ($n_{latents} = n_{embeddings}$)"),
        (ax_right, "parameter-matched"),
    ):
        ax.set_xlabel("$L_0$")
        ax.set_title(subtitle)
        ax.legend()

    ax_left.set_ylabel("Recon MSE")
    fig.suptitle(
        f"{ds_cfg.name} — L0 vs Reconstruction MSE\n"
        f"VAEE hidden_dim={cfg.vaee_hidden_dim}, "
        f"embedding_size={cfg.vaee_embedding_size}, "
        f"SAE $\\lambda_{{L_1}}$={cfg.sae_lambda_l1}",
        fontsize=18,
    )
    fig.tight_layout()
    path = out_dir / "l0_recon_scatter.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path.name}")


def plot_learning_curves(
    results: list[RunResult],
    cfg: RunConfig,
    out_dir: Path,
) -> None:
    """Grid of learning curves, one subplot per (model, n_concepts).

    Each subplot shows train and validation reconstruction MSE over epochs,
    with a vertical dotted line at the best-checkpoint epoch.

    Args:
        results: Collected RunResult objects from run_experiment().
        cfg: Run hyperparameters (used for layout).
        out_dir: Directory to save the figure into.

    """
    present_models = [m for m in MODEL_NAMES if any(r.model_name == m for r in results)]
    n_rows = len(present_models)
    n_cols = len(cfg.num_embeddings_list)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    epochs = range(1, cfg.epochs + 1)

    for row, model_name in enumerate(present_models):
        for col, n in enumerate(cfg.num_embeddings_list):
            ax = axes[row][col]
            rs = [r for r in results if r.model_name == model_name and r.sweep_n == n]
            if not rs:
                ax.set_visible(False)
                continue
            r = rs[0]
            best_epoch = int(r.val_recon.index(min(r.val_recon))) + 1
            color = COLORS[model_name]
            ax.plot(epochs, r.train_recon, lw=1, color=color, label="train")
            ax.plot(epochs, r.val_recon, lw=2, color=color, label="val")
            ax.axvline(best_epoch, ls="--", color=color, lw=1, alpha=0.7)
            if row == 0:
                ax.set_title(f"n_latents={n}")
            if col == 0:
                ax.set_ylabel(f"{model_name}\nRecon MSE")
                ax.legend()
            if row == n_rows - 1:
                ax.set_xlabel("Epoch")

    fig.suptitle("Learning curves", fontsize=18)
    fig.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path.name}")

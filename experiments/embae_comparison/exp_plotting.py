# ruff: noqa: N803, N806

"""Plotting utilities for the SparseAE vs EmbeddingAE comparison."""

from __future__ import annotations

import matplotlib as mpl
import torch

mpl.use("Agg")
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .exp_data import denormalize

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.preprocessing import StandardScaler
    from torch import Tensor, nn

    from .exp_config import DatasetConfig, RunConfig
    from .exp_training import RunResult

COLORS = {"SparseAE": "steelblue", "EmbeddingAE": "coral"}
MARKERS = {"SparseAE": "o", "EmbeddingAE": "^"}


def plot_l0_recon(
    results: list[RunResult],
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
    out_dir: Path,
) -> None:
    """Scatter plot of L0 vs reconstruction MSE, one point per (model, n_concepts).

    Args:
        results: Collected RunResult objects from run_experiment().
        cfg: Run hyperparameters (used for axis annotations).
        ds_cfg: Dataset metadata (used for the figure title).
        out_dir: Directory to save the figure into.

    """
    fig, ax = plt.subplots()

    for model_name in ("SparseAE", "EmbeddingAE"):
        rs = [r for r in results if r.model_name == model_name]
        l0s = [r.best_l0 for r in rs]
        recons = [r.best_test_recon for r in rs]
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

    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("$L_0$")
    ax.set_ylabel("Recon MSE")

    if cfg.sparsity_mode == "l1":
        sparsity_label = r"$\lambda_{L_1}=$" + f"${cfg.lambda_l1}$"
    else:
        sparsity_label = r"$\lambda_{KL}=$" + f"${cfg.lambda_kl}$, $p^*={cfg.target_p}$"

    ax.set_title(
        f"{ds_cfg.name} — L0 vs Reconstruction Loss"
        " (point annotations = n_concepts)\n"
        f"{sparsity_label}, EmbeddingAE embedding_size={cfg.embedding_size}",
    )
    ax.legend()
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

    Each subplot shows train and test reconstruction MSE over epochs, with a
    vertical dotted line marking the best checkpoint epoch.

    Args:
        results: Collected RunResult objects from run_experiment().
        cfg: Run hyperparameters (used for axis layout and annotations).
        out_dir: Directory to save the figure into.

    """
    n_rows, n_cols = 2, len(cfg.n_concepts_list)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.5 * n_rows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    model_names = ["SparseAE", "EmbeddingAE"]
    epochs = range(1, cfg.epochs + 1)

    for row, model_name in enumerate(model_names):
        for col, nc in enumerate(cfg.n_concepts_list):
            ax = axes[row][col]
            r = next(
                (
                    r
                    for r in results
                    if r.model_name == model_name and r.n_concepts == nc
                ),
                None,
            )
            if r is None:
                ax.axis("off")
                continue
            c = COLORS[model_name]
            ax.plot(epochs, r.train_recon, color=c, alpha=0.5, label="train")
            ax.plot(epochs, r.test_recon, color=c, label="test")
            best_epoch = int(np.argmin(r.test_recon)) + 1
            ax.axvline(best_epoch, color=c, linewidth=1, linestyle=":", alpha=0.7)
            # ax.set_yscale("log")
            ax.set_title(f"{model_name} n={nc}")
            if col == 0:
                if row == 1:
                    ax.set_xlabel("Epoch")
                ax.set_ylabel("Recon MSE")
                legend = ax.legend()
            ax.text(
                0.97,
                0.95,
                f"$L_0={r.best_l0:.1f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=c,
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": legend.get_frame().get_facecolor()
                    if legend
                    else "white",
                    "edgecolor": legend.get_frame().get_edgecolor() if legend else c,
                    "linewidth": legend.get_frame().get_linewidth() if legend else 0.8,
                    "alpha": legend.get_frame().get_alpha() or 0.8 if legend else 0.8,
                },
            )

    fig.suptitle("Learning Curves", fontsize=16)
    fig.tight_layout()
    path = out_dir / "learning_curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def plot_concept_dictionary(  # noqa: PLR0913
    model: nn.Module,
    model_name: str,
    n_concepts: int,
    X_train: Tensor,
    scaler: StandardScaler,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
    out_dir: Path,
) -> None:
    """Image grid of top-activating examples per concept.

    Concepts are sorted by total activation mass and capped at cfg.max_concepts_in_dict.
    For each concept, cfg.top_k_examples images are shown in a column.

    Args:
        model: Trained model (best checkpoint).
        model_name: "SparseAE" or "EmbeddingAE".
        n_concepts: Number of concepts the model was trained with.
        X_train: Normalised training tensor used to compute activations.
        scaler: Fitted StandardScaler for denormalisation before display.
        cfg: Run hyperparameters (top_k_examples, max_concepts_in_dict, device).
        ds_cfg: Dataset metadata (img_shape, img_vmax for correct display).
        out_dir: Directory to save the figure into.

    """
    model.eval()
    with torch.inference_mode():
        out = model(X_train.to(cfg.device))

    # Use raw (non-binarised) activations for ranking
    activations: np.ndarray
    if model_name == "SparseAE":
        activations = out.latents.cpu().numpy()  # (N, n_concepts)
    else:
        activations = out.scores.cpu().numpy()  # (N, n_concepts), sigmoid ∈ (0,1)

    concept_total_act = activations.sum(axis=0)
    top_concept_idxs = np.argsort(concept_total_act)[::-1][: cfg.max_concepts_in_dict]
    n_show = len(top_concept_idxs)

    X_np = X_train.numpy()
    nrows, ncols = cfg.top_k_examples, n_show
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.3))

    if nrows == 1:
        axes = axes[np.newaxis, :]

    for col, concept_idx in enumerate(top_concept_idxs):
        act = activations[:, concept_idx]
        top_ex_idxs = np.argsort(act)[::-1][: cfg.top_k_examples]

        for row, ex_idx in enumerate(top_ex_idxs):
            ax = axes[row, col]
            img = denormalize(X_np[ex_idx : ex_idx + 1], scaler)[0].reshape(
                ds_cfg.img_shape,
            )
            img = np.clip(img, 0, ds_cfg.img_vmax)
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=ds_cfg.img_vmax)
            if row == 0:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_title(f"C{concept_idx}", rotation=0, pad=15, va="center")
            else:
                ax.axis("off")

    title = f"{model_name} — concept dictionary ({n_concepts} concepts)"
    if n_concepts > cfg.max_concepts_in_dict:
        title += f"\nShowing {n_show} most-active concepts out of {n_concepts}"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    path = out_dir / f"concept_dict_{model_name.lower()}_{n_concepts:03d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")

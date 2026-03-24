# ruff: noqa: N803

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


def _decode_prototype(
    model: nn.Module,
    model_name: str,
    concept_idx: int,
    n_concepts: int,
    device: torch.device,
) -> np.ndarray:
    """Decode the prototype for concept_idx in normalized space.

    For EmbeddingAE: passes the learned prototypes through the decoder with a
    one-hot score vector, isolating concept_idx.
    For SparseAE: decodes a one-hot latent using the mean LayerNorm params
    from the training forward pass, giving a representative prototype image.

    Returns:
        Array of shape (1, input_dim) in globally-normalized space.

    """
    with torch.inference_mode():
        if model_name == "EmbeddingAE":
            prototypes = model.prototypes.unsqueeze(
                0,
            )  # (1, num_embeddings, embedding_size)
            scores = torch.zeros(1, n_concepts, device=device)
            scores[0, concept_idx] = 1.0
            recon = model.decode(prototypes, scores)
        elif model_name == "SparseAE":  # SparseAE
            z = torch.zeros(1, n_concepts, device=device)
            z[0, concept_idx] = 100.0
            recon = model.decode(z)
        else:
            msg = "Model not supported"
            raise ValueError(msg)
    return recon.cpu().numpy()  # (1, input_dim)


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

    fig.suptitle("Learning Curves", fontsize=20)
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
    For each concept, cfg.top_k_examples images are shown in a column. An additional row
    shows the decoder reconstruction of each latent/prototype in isolation (one-hot
    scores).

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
        activations = out.scores.cpu().numpy()  # (N, n_concepts)

    concept_total_act = activations.sum(axis=0)
    top_concept_idxs = np.argsort(concept_total_act)[::-1][: cfg.max_concepts_in_dict]
    n_show = len(top_concept_idxs)

    ncols = n_show
    # Row layout: 1 prototype row, 1 invisible gap row, top_k_examples example rows
    n_example_rows = cfg.top_k_examples
    height_ratios = [1.0, 0.15] + [1.0] * n_example_rows
    fig = plt.figure(figsize=(ncols * 1.3, (n_example_rows + 1.15) * 1.3))
    gs = fig.add_gridspec(
        n_example_rows + 2,
        ncols,
        height_ratios=height_ratios,
        hspace=0.05,
        wspace=0.05,
    )

    for col, concept_idx in enumerate(top_concept_idxs):
        act = activations[:, concept_idx]
        top_ex_idxs = np.argsort(act)[::-1][: cfg.top_k_examples]

        # ── Prototype row (row 0) ─────────────────────────────────────────────
        proto_np = _decode_prototype(
            model,
            model_name,
            concept_idx,
            n_concepts,
            cfg.device,
        )
        proto_img = denormalize(proto_np, scaler)[0].reshape(ds_cfg.img_shape)
        proto_img = np.clip(proto_img, 0, ds_cfg.img_vmax)
        ax_proto = fig.add_subplot(gs[0, col])
        ax_proto.imshow(proto_img, cmap="gray_r", vmin=0, vmax=ds_cfg.img_vmax)
        ax_proto.set_xticks([])
        ax_proto.set_yticks([])
        for spine in ax_proto.spines.values():
            spine.set_visible(False)
        ax_proto.set_title(f"C{concept_idx}", rotation=0, pad=15, va="center")
        if col == 0:
            ax_proto.set_ylabel(
                "prototype",
                fontsize=11,
                va="center",
            )

        # ── Gap row (row 1) — invisible ───────────────────────────────────────
        ax_gap = fig.add_subplot(gs[1, col])
        ax_gap.axis("off")

        # ── Example rows (rows 2..) ───────────────────────────────────────────
        for row, ex_idx in enumerate(top_ex_idxs):
            img = denormalize(out.recon[ex_idx : ex_idx + 1].cpu().numpy(), scaler)[
                0
            ].reshape(
                ds_cfg.img_shape,
            )
            img = np.clip(img, 0, ds_cfg.img_vmax)
            ax = fig.add_subplot(gs[row + 2, col])
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=ds_cfg.img_vmax)
            ax.axis("off")

    title = f"{model_name} — concept dictionary ({n_concepts} concepts)"
    if n_concepts > cfg.max_concepts_in_dict:
        title += f"\nShowing {n_show} most-active concepts out of {n_concepts}"
    fig.suptitle(title, fontsize=20)
    gs.tight_layout(fig)

    path = out_dir / f"concept_dict_{model_name.lower()}_{n_concepts:03d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _ablated_recon(
    model: nn.Module,
    model_name: str,
    x: Tensor,
    concept_idx: int,
    device: torch.device,
) -> np.ndarray:
    """Return the reconstruction of x with concept_idx zeroed out."""
    with torch.inference_mode():
        out = model(x.to(device))
        if model_name == "SparseAE":
            z = out.latents.clone()
            z[:, concept_idx] = 0.0
            recon = model.decode(z)
        elif model_name == "EmbeddingAE":
            scores = out.scores.clone()
            scores[:, concept_idx] = 0.0
            recon = model.decode(out.embeddings, scores)
        else:
            msg = "Model not supported"
            raise ValueError(msg)

    return recon.cpu().numpy()


def plot_concept_ablation(  # noqa: PLR0913, PLR0915
    model: nn.Module,
    model_name: str,
    n_concepts: int,
    X_train: Tensor,
    scaler: StandardScaler,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
    out_dir: Path,
) -> None:
    """Image grid showing decoded prototype and top-activating examples alongside their ablated reconstructions.

    Layout per concept group (2 columns: orig | ablated):
        row 0:   prototype image, spanning both columns, titled "CX"
        row 1:   "orig" subtitle | "ablated" subtitle  (no images)
        row 2:   (gap)
        rows 3+: top-activating example | ablated reconstruction

    Args:
        model: Trained model (best checkpoint).
        model_name: "SparseAE" or "EmbeddingAE".
        n_concepts: Number of concepts the model was trained with.
        X_train: Normalised training tensor.
        scaler: Fitted StandardScaler for denormalisation before display.
        cfg: Run hyperparameters.
        ds_cfg: Dataset metadata.
        out_dir: Directory to save the figure into.

    """  # noqa: E501
    model.eval()
    with torch.no_grad():
        out = model(X_train.to(cfg.device))

    activations: np.ndarray
    if model_name == "SparseAE":
        activations = out.latents.cpu().numpy()
    else:
        activations = out.scores.cpu().numpy()

    concept_total_act = activations.sum(axis=0)
    top_concept_idxs = np.argsort(concept_total_act)[::-1][: cfg.max_concepts_in_dict]
    n_show = len(top_concept_idxs)

    n_example_rows = cfg.top_k_examples
    ncols = n_show * 2  # orig | ablated per concept
    # Row layout: prototype, subtitle, gap, examples
    height_ratios = [1.0, 0.2] + [
        1.0,
    ] * n_example_rows  # [prototype, orig/ablated] + [examples]
    fig = plt.figure(figsize=(ncols * 0.9, (n_example_rows + 1.3) * 1.3))
    gs = fig.add_gridspec(
        n_example_rows + 2,
        ncols,
        height_ratios=height_ratios,
        hspace=0.05,
        wspace=0.05,
    )

    for concept_col, concept_idx in enumerate(top_concept_idxs):
        act = activations[:, concept_idx]
        top_ex_idxs = np.argsort(act)[::-1][: cfg.top_k_examples]
        orig_col = concept_col * 2
        ablated_col = orig_col + 1

        # ── Prototype row (row 0): spans both columns ─────────────────────────
        proto_np = _decode_prototype(
            model,
            model_name,
            concept_idx,
            n_concepts,
            cfg.device,
        )
        proto_img = denormalize(proto_np, scaler)[0].reshape(ds_cfg.img_shape)
        proto_img = np.clip(proto_img, 0, ds_cfg.img_vmax)
        ax_proto = fig.add_subplot(gs[0, orig_col : ablated_col + 1])
        ax_proto.imshow(proto_img, cmap="gray_r", vmin=0, vmax=ds_cfg.img_vmax)
        ax_proto.set_xticks([])
        ax_proto.set_yticks([])
        for spine in ax_proto.spines.values():
            spine.set_visible(False)
        ax_proto.set_title(f"C{concept_idx}", fontsize=16, pad=4)
        if concept_col == 0:
            ax_proto.set_ylabel(
                "prototype",
                fontsize=11,
                labelpad=22,
                va="center",
            )

        # ── Subtitle row (row 1): "orig" | "ablated" side by side ─────────────
        for col_idx, label in ((orig_col, "orig"), (ablated_col, "ablated")):
            ax_sub = fig.add_subplot(gs[1, col_idx])
            ax_sub.axis("off")
            ax_sub.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_sub.transAxes,
            )

        # ── Example rows (rows 2+) ────────────────────────────────────────────
        for row, ex_idx in enumerate(top_ex_idxs):
            x_single = X_train[ex_idx : ex_idx + 1]

            img_orig = denormalize(
                out.recon[ex_idx : ex_idx + 1].cpu().numpy(),
                scaler,
            )[0].reshape(ds_cfg.img_shape)
            recon_np = _ablated_recon(
                model,
                model_name,
                x_single,
                concept_idx,
                cfg.device,
            )
            recon_np = denormalize(recon_np, scaler)
            img_ablated = np.clip(
                recon_np[0].reshape(ds_cfg.img_shape),
                0,
                ds_cfg.img_vmax,
            )

            for col_idx, img in ((orig_col, img_orig), (ablated_col, img_ablated)):
                ax = fig.add_subplot(gs[row + 2, col_idx])
                ax.imshow(img, cmap="gray_r", vmin=0, vmax=ds_cfg.img_vmax)
                ax.axis("off")

    title = f"{model_name} — concept ablation ({n_concepts} concepts)"
    if n_concepts > cfg.max_concepts_in_dict:
        title += f"\nShowing {n_show} most-active concepts out of {n_concepts}"
    fig.suptitle(title, fontsize=16)
    gs.tight_layout(fig)

    path = out_dir / f"concept_ablation_{model_name.lower()}_{n_concepts:03d}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")

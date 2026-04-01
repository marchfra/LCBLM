# ruff: noqa: E402, N806
"""EmbeddingAE — Concept Quality Diagnosis.

Investigates *why* EmbeddingAE reconstructs better (lower MSE, lower L0) than SparseAE
but produces worse concept dictionaries.

Hypothesis: prototypes are used only as gate references (inner-product alignment → score);
the decoder reconstructs from input-specific encoder embeddings and was never trained on
prototypes directly, so prototypes carry no pixel-space meaning.

Diagnostics:
1. Inner-product alignment distribution (active vs inactive concepts)
2. Prototype decoded vs. average-active-embedding decoded — side-by-side image grid
3. Within-concept encoder embedding variance
4. Per-concept prototype-embedding distance
"""  # noqa: E501

# ── Parameters ────────────────────────────────────────────────────────────────
# Point this at any run directory that contains results.json + checkpoints/
# Can be overridden via CLI: python 01_embae_diagnosis.py <run_dir> [n_concepts]
RUN_DIR = "/Users/francescomarchisotti/Documents/Uni/MasterThesis/code/LCBLM/experiments/embae_comparison/experiment_outputs/run_052/"  # noqa: E501

# Which n_concepts model to load (must be present in results.json)
N_CONCEPTS = 10

# Score threshold for "active" (GumbelSigmoid outputs hard 0/1 in eval, so 0.5 works)
ACTIVE_THRESHOLD = 0.5

# How many top concepts to show in the image grids
MAX_CONCEPTS_SHOW = 10

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sure the repo root is on the path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.embae_comparison.exp_cli import (
    _build_model,
    _checkpoint_path,
    _scaler_path,
)
from experiments.embae_comparison.exp_data import DATASET_REGISTRY, denormalize
from experiments.embae_comparison.exp_io import load_results

plt.switch_backend("Agg")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default=RUN_DIR)
    parser.add_argument("n_concepts", nargs="?", type=int, default=N_CONCEPTS)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    _n_concepts = args.n_concepts

    # Output directory named after the run so multiple runs don't overwrite each other
    out_dir = run_dir / "diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs → {out_dir}")
    device = get_device()
    print(f"Device: {device}")

    results, run_cfg, ds_cfg = load_results(run_dir / "results.json")
    print(f"Dataset : {ds_cfg.name}  input_dim={ds_cfg.input_dim}")
    print(
        f"Run cfg : epochs={run_cfg.epochs}  lr={run_cfg.lr}  sparsity={run_cfg.sparsity_mode}",  # noqa: E501
    )
    print(f"Results : {len(results)} run(qs)")

    with _scaler_path(run_dir).open("rb") as f:
        scaler = pickle.load(f)  # noqa: S301

    _, load_data = DATASET_REGISTRY[ds_cfg.name]
    X_train, X_test, _y_train, _y_test, _ = load_data(ds_cfg.n_samples)
    print(f"\nX_train: {X_train.shape}  X_test: {X_test.shape}")

    # ── Load EmbeddingAE and SparseAE checkpoints for N_CONCEPTS ─────────────────
    models = {}
    for model_name in ("EmbeddingAE", "SparseAE"):
        ckpt = _checkpoint_path(run_dir, model_name, N_CONCEPTS)
        if not ckpt.exists():
            print(
                f"WARNING: checkpoint not found for {model_name} n={N_CONCEPTS}: {ckpt}",  # noqa: E501
            )
            continue
        m = _build_model(model_name, N_CONCEPTS, run_cfg, ds_cfg)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        models[model_name] = m
        print(f"Loaded {model_name}  (n_concepts={N_CONCEPTS})")

    embae = models["EmbeddingAE"]
    sparseae = models.get("SparseAE")

    # ── Forward pass (eval → GumbelSigmoid outputs hard 0/1) ─────────────────────
    X = X_train.to(device)

    with torch.inference_mode():
        emb_out = embae(X)

    embeddings = emb_out.embeddings.cpu()  # (N, n_concepts, embed_size)
    scores = emb_out.scores.cpu()  # (N, n_concepts)  — hard {0,1} in eval
    alignments = emb_out.alignments.cpu()  # (N, n_concepts)  — raw inner products
    _recon_emb = emb_out.recon.cpu()  # (N, input_dim)

    active_mask = scores > ACTIVE_THRESHOLD  # (N, n_concepts)  boolean
    active_rate = active_mask.float().mean(dim=0)  # (n_concepts,)

    print(f"Mean active concepts per sample : {active_mask.float().sum(-1).mean():.2f}")
    print(
        f"Concepts with >0 active samples : {(active_rate > 0).sum().item()} / {N_CONCEPTS}",  # noqa: E501
    )

    if sparseae is not None:
        with torch.inference_mode():
            sae_out = sparseae(X)
        sae_latents = sae_out.latents.cpu()  # (N, n_concepts)
        sae_pre_activation = sae_out.latents_pre_activation.cpu()  # (N, n_concepts)
        _sae_recon = sae_out.recon.cpu()  # (N, input_dim)
        sae_active_mask = sae_latents > 0  # (N, n_concepts)
        sae_active_rate = sae_active_mask.float().mean(dim=0)
        print(
            f"\n[SparseAE] Mean active concepts per sample : {sae_active_mask.float().sum(-1).mean():.2f}",  # noqa: E501
        )
        print(
            f"[SparseAE] Concepts with >0 active samples : {(sae_active_rate > 0).sum().item()} / {N_CONCEPTS}",  # noqa: E501
        )

    # ── Diagnostic 1 — Inner-product alignment distributions ─────────────────────
    print("\n=== Diagnostic 1: Inner-product alignment distributions ===")

    # Sort concepts by total activation mass (most active first)
    total_act = active_mask.float().sum(dim=0).numpy()  # (n_concepts,)
    sorted_concept_idxs = np.argsort(total_act)[::-1][:MAX_CONCEPTS_SHOW]

    n_show = len(sorted_concept_idxs)
    ncols = 5
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3, nrows * 2.5),
        sharex=False,
        sharey=False,
    )
    axes = np.array(axes).flatten()

    for plot_i, concept_i in enumerate(sorted_concept_idxs):
        ax = axes[plot_i]
        is_active = active_mask[:, concept_i].numpy()
        align_vals = alignments[:, concept_i].numpy()

        bins = 40
        ax.hist(
            align_vals[is_active],
            bins=bins,
            alpha=0.7,
            label="active",
            color="coral",
            density=True,
        )
        ax.hist(
            align_vals[~is_active],
            bins=bins,
            alpha=0.7,
            label="inactive",
            color="steelblue",
            density=True,
        )

        # Mark the mean of each group
        if is_active.sum() > 0:
            ax.axvline(align_vals[is_active].mean(), color="firebrick", ls="--", lw=1.5)
        if (~is_active).sum() > 0:
            ax.axvline(align_vals[~is_active].mean(), color="navy", ls="--", lw=1.5)

        n_act = int(is_active.sum())
        ax.set_title(f"C{concept_i}  (n_active={n_act})", fontsize=9)
        if plot_i == 0:
            ax.legend(fontsize=8)
        ax.set_xlabel("⟨emb_i, proto_i⟩", fontsize=8)

    for ax in axes[n_show:]:
        ax.axis("off")

    fig.suptitle(
        f"EmbeddingAE — inner-product alignment distributions (n_concepts={N_CONCEPTS})",  # noqa: E501
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "diag1_alignment_distributions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved diag1_alignment_distributions.png")

    # ── Diagnostic 2 — Prototype decoded vs. average active encoder embedding decoded ──  # noqa: E501
    print("\n=== Diagnostic 2: Prototype vs avg-active-embedding decoded ===")

    def decode_single_concept(
        model,  # noqa: ANN001
        concept_embedding: torch.Tensor,  # (embed_size,)
        concept_idx: int,
        n_concepts: int,
        device: torch.device,
    ) -> np.ndarray:
        """Decode a single concept embedding in isolation (one-hot score).

        Builds a (1, n_concepts, embed_size) tensor with only concept_idx filled
        and passes it through the decoder with a one-hot score.
        """
        embed_size = concept_embedding.shape[-1]
        fake_embeds = torch.zeros(1, n_concepts, embed_size, device=device)
        fake_embeds[0, concept_idx] = concept_embedding.to(device)
        fake_scores = torch.zeros(1, n_concepts, device=device)
        fake_scores[0, concept_idx] = 1.0
        with torch.inference_mode():
            recon = model.decode(fake_embeds, fake_scores)
        return recon.cpu().numpy()  # (1, input_dim)

    prototypes = embae.prototypes.detach().cpu()  # (n_concepts, embed_size)

    n_show = len(sorted_concept_idxs)
    fig, axes = plt.subplots(
        2,
        n_show,
        figsize=(n_show * 1.6, 2 * 1.6 + 0.5),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    row_labels = ["prototype", "avg\nactive emb"]

    for col, concept_i in enumerate(sorted_concept_idxs):
        is_active_i = active_mask[:, concept_i].numpy()
        n_act = int(is_active_i.sum())

        # ── Row 0: decode raw prototype ───────────────────────────────────────────
        proto_img = decode_single_concept(
            embae,
            prototypes[concept_i],
            concept_i,
            N_CONCEPTS,
            device,
        )
        proto_img = denormalize(proto_img, scaler)[0].reshape(ds_cfg.img_shape)
        proto_img = np.clip(proto_img, 0, ds_cfg.img_vmax)

        # ── Row 1: decode mean active encoder embedding ───────────────────────────
        if n_act > 0:
            active_embeds_i = embeddings[
                is_active_i,
                concept_i,
                :,
            ]  # (n_act, embed_size)
            mean_embed_i = active_embeds_i.mean(dim=0)  # (embed_size,)
            avg_img = decode_single_concept(
                embae,
                mean_embed_i,
                concept_i,
                N_CONCEPTS,
                device,
            )
            avg_img = denormalize(avg_img, scaler)[0].reshape(ds_cfg.img_shape)
            avg_img = np.clip(avg_img, 0, ds_cfg.img_vmax)
        else:
            avg_img = np.zeros(ds_cfg.img_shape)

        for row, img in enumerate([proto_img, avg_img]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=ds_cfg.img_vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"C{concept_i}\n(n={n_act})", fontsize=9)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, labelpad=4)

    fig.suptitle(
        f"EmbeddingAE — prototype vs avg-active-embedding decoded (n_concepts={N_CONCEPTS})",  # noqa: E501
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "diag2_prototype_vs_avg_embedding.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved diag2_prototype_vs_avg_embedding.png")

    # ── Diagnostic 3 — Within-concept encoder embedding variance ───────────────────
    print("\n=== Diagnostic 3: Within-concept encoder embedding variance ===")

    within_concept_var = np.zeros(N_CONCEPTS)
    n_active_per_concept = active_mask.float().sum(dim=0).numpy()

    for i in range(N_CONCEPTS):
        mask_i = active_mask[:, i].numpy().astype(bool)
        if (
            mask_i.sum() < 2  # noqa: PLR2004
        ):  # need at least 2 samples to compute variance
            within_concept_var[i] = 0.0
            continue
        active_embeds = embeddings[mask_i, i, :].numpy()  # (n_act, embed_size)
        within_concept_var[i] = active_embeds.var(axis=0).mean()  # scalar

    # Sort by activation frequency for readability
    order = np.argsort(n_active_per_concept)[::-1]

    fig, ax = plt.subplots(figsize=(max(8, N_CONCEPTS * 0.55), 4))
    x = np.arange(N_CONCEPTS)
    _bars = ax.bar(x, within_concept_var[order], color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{i}\n(n={int(n_active_per_concept[i])})" for i in order],
        fontsize=8,
        rotation=0,
    )
    ax.set_ylabel("Mean embedding variance\n(over active samples)")
    ax.set_title(
        f"EmbeddingAE — within-concept encoder embedding variance (n_concepts={N_CONCEPTS})\n"  # noqa: E501
        "High = encoder produces different embeddings for the same concept across samples",  # noqa: E501
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "diag3_within_concept_variance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved diag3_within_concept_variance.png")

    print(
        f"Mean within-concept variance across all concepts: {within_concept_var.mean():.4f}",  # noqa: E501
    )
    print(
        f"Max:  {within_concept_var.max():.4f}   Min (active): "
        f"{within_concept_var[within_concept_var > 0].min():.4f}",
    )

    # ── Diagnostic 4 — Per-concept prototype-embedding distance ─────────────────────
    print("\n=== Diagnostic 4: Per-concept prototype-embedding distance ===")

    proto_embed_dist = np.zeros(N_CONCEPTS)
    proto_embed_dist_normalized = np.zeros(N_CONCEPTS)  # distance / ||prototype||

    for i in range(N_CONCEPTS):
        proto_i = prototypes[i].numpy()  # (embed_size,)
        mask_i = active_mask[:, i].numpy().astype(bool)
        if mask_i.sum() == 0:
            proto_embed_dist[i] = np.nan
            proto_embed_dist_normalized[i] = np.nan
            continue
        mean_embed_i = embeddings[mask_i, i, :].numpy().mean(axis=0)  # (embed_size,)
        dist = np.linalg.norm(proto_i - mean_embed_i)
        proto_embed_dist[i] = dist
        proto_embed_dist_normalized[i] = dist / (np.linalg.norm(proto_i) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, vals, ylabel, title_suffix in zip(
        axes,
        [proto_embed_dist, proto_embed_dist_normalized],
        ["||proto_i - mean_embed_i||₂", "distance / ||proto_i||₂"],
        ["absolute", "relative to prototype norm"],
        strict=True,
    ):
        valid = ~np.isnan(vals[order])
        ax.bar(x[valid], vals[order][valid], color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"C{i}\n(n={int(n_active_per_concept[i])})" for i in order],
            fontsize=8,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(f"Prototype-avg-active-embedding distance ({title_suffix})")

    fig.suptitle(f"EmbeddingAE  n_concepts={N_CONCEPTS}", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        out_dir / "diag4_prototype_embedding_distance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved diag4_prototype_embedding_distance.png")

    valid_mask = ~np.isnan(proto_embed_dist)
    print("Prototype-embedding distances (active concepts only):")
    print(f"  Mean absolute : {proto_embed_dist[valid_mask].mean():.4f}")
    print(f"  Mean relative : {proto_embed_dist_normalized[valid_mask].mean():.4f}")
    print()
    print("Prototype norms:")
    proto_norms = prototypes.norm(dim=-1).numpy()
    print(
        f"  Mean={proto_norms.mean():.3f}  Std={proto_norms.std():.3f}  "
        f"Min={proto_norms.min():.3f}  Max={proto_norms.max():.3f}",
    )
    print("Active embedding norms:")
    embed_norms = embeddings.norm(dim=-1).numpy()  # (N, n_concepts)
    active_embed_norms = embed_norms[active_mask.numpy()]
    print(f"  Mean={active_embed_norms.mean():.3f}  Std={active_embed_norms.std():.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # ── SparseAE diagnostics (comparison baseline) ───────────────────────────
    # ═══════════════════════════════════════════════════════════════════════════
    if sparseae is not None:
        X_np = X.cpu().numpy()  # (N, input_dim)  — normalised pixel space

        sae_total_act = sae_active_mask.float().sum(dim=0).numpy()
        sae_sorted_idxs = np.argsort(sae_total_act)[::-1][:MAX_CONCEPTS_SHOW]
        n_show_sae = len(sae_sorted_idxs)

        # ── SAE Diag 1 — Pre-activation latent distributions ─────────────────
        print("\n=== [SparseAE] Diagnostic 1: Pre-activation latent distributions ===")

        ncols_s = 5
        nrows_s = int(np.ceil(n_show_sae / ncols_s))
        fig, axes = plt.subplots(
            nrows_s,
            ncols_s,
            figsize=(ncols_s * 3, nrows_s * 2.5),
            sharex=False,
            sharey=False,
        )
        axes = np.array(axes).flatten()

        for plot_i, concept_i in enumerate(sae_sorted_idxs):
            ax = axes[plot_i]
            is_act_i = sae_active_mask[:, concept_i].numpy()
            pre_act_i = sae_pre_activation[:, concept_i].numpy()

            bins = 40
            ax.hist(
                pre_act_i[is_act_i],
                bins=bins,
                alpha=0.7,
                label="active",
                color="coral",
                density=True,
            )
            ax.hist(
                pre_act_i[~is_act_i],
                bins=bins,
                alpha=0.7,
                label="inactive",
                color="steelblue",
                density=True,
            )
            if is_act_i.sum() > 0:
                ax.axvline(
                    pre_act_i[is_act_i].mean(),
                    color="firebrick",
                    ls="--",
                    lw=1.5,
                )
            if (~is_act_i).sum() > 0:
                ax.axvline(pre_act_i[~is_act_i].mean(), color="navy", ls="--", lw=1.5)

            ax.set_title(f"C{concept_i}  (n_active={int(is_act_i.sum())})", fontsize=9)
            if plot_i == 0:
                ax.legend(fontsize=8)
            ax.set_xlabel("pre-activation latent", fontsize=8)

        for ax in axes[n_show_sae:]:
            ax.axis("off")

        fig.suptitle(
            f"SparseAE — pre-activation latent distributions (n_concepts={N_CONCEPTS})",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / "sae_diag1_preact_distributions.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print("Saved sae_diag1_preact_distributions.png")

        # ── SAE Diag 2 — Decoded mean-latent vs mean active input ─────────
        print(
            "\n=== [SparseAE] Diagnostic 2: Decoded mean-latent vs mean active input ===",  # noqa: E501
        )

        fig, axes = plt.subplots(
            2,
            n_show_sae,
            figsize=(n_show_sae * 1.6, 2 * 1.6 + 0.5),
            gridspec_kw={"hspace": 0.05, "wspace": 0.05},
        )

        for col, concept_i in enumerate(sae_sorted_idxs):
            is_act_i = sae_active_mask[:, concept_i].numpy()
            n_act = int(is_act_i.sum())

            # Row 0: decode at mean active latent value (the SAE "prototype")
            is_act_i_bool = sae_active_mask[:, concept_i].numpy()
            mean_latent_val = (
                float(sae_latents[is_act_i_bool, concept_i].mean())
                if is_act_i_bool.sum() > 0
                else 1.0
            )
            z = torch.zeros(1, N_CONCEPTS, device=device)
            z[0, concept_i] = mean_latent_val
            with torch.inference_mode():
                proto_recon = sparseae.decode(z).cpu().numpy()  # (1, input_dim)
            proto_img = denormalize(proto_recon, scaler)[0].reshape(ds_cfg.img_shape)
            proto_img = np.clip(proto_img, 0, ds_cfg.img_vmax)

            # Row 1: mean of raw inputs when concept is active (pixel space)
            if n_act > 0:
                mean_input = X_np[is_act_i].mean(
                    axis=0,
                    keepdims=True,
                )  # (1, input_dim)
                avg_img = denormalize(mean_input, scaler)[0].reshape(ds_cfg.img_shape)
                avg_img = np.clip(avg_img, 0, ds_cfg.img_vmax)
            else:
                avg_img = np.zeros(ds_cfg.img_shape)

            for row, img in enumerate([proto_img, avg_img]):
                ax = axes[row, col]
                ax.imshow(img, cmap="gray_r", vmin=0, vmax=ds_cfg.img_vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(f"C{concept_i}\n(n={n_act})", fontsize=9)
                if col == 0:
                    ax.set_ylabel(
                        ["decoded\nmean latent", "mean\nactive input"][row],
                        fontsize=9,
                        labelpad=4,
                    )

        fig.suptitle(
            f"SparseAE — decoded mean-latent vs mean active input (n_concepts={N_CONCEPTS})",  # noqa: E501
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / "sae_diag2_prototype_vs_mean_input.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print("Saved sae_diag2_prototype_vs_mean_input.png")

        # ── SAE Diag 3 — Within-concept input variance ────────────────────────
        print("\n=== [SparseAE] Diagnostic 3: Within-concept input variance ===")

        sae_within_input_var = np.zeros(N_CONCEPTS)
        for i in range(N_CONCEPTS):
            mask_i = sae_active_mask[:, i].numpy().astype(bool)
            if mask_i.sum() < 2:  # noqa: PLR2004
                sae_within_input_var[i] = 0.0
                continue
            sae_within_input_var[i] = X_np[mask_i].var(axis=0).mean()

        sae_order = np.argsort(sae_total_act)[::-1]
        x_s = np.arange(N_CONCEPTS)
        fig, ax = plt.subplots(figsize=(max(8, N_CONCEPTS * 0.55), 4))
        ax.bar(x_s, sae_within_input_var[sae_order], color="steelblue", alpha=0.8)
        ax.set_xticks(x_s)
        ax.set_xticklabels(
            [f"C{i}\n(n={int(sae_total_act[i])})" for i in sae_order],
            fontsize=8,
        )
        ax.set_ylabel("Mean input variance\n(over active samples)")
        ax.set_title(
            f"SparseAE — within-concept input variance (n_concepts={N_CONCEPTS})\n"
            "High = inputs when concept fires are very diverse",
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / "sae_diag3_within_concept_input_variance.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print("Saved sae_diag3_within_concept_input_variance.png")

        print(f"Mean within-concept input variance : {sae_within_input_var.mean():.4f}")
        active_vars = sae_within_input_var[sae_within_input_var > 0]
        if len(active_vars):
            print(
                f"Max: {active_vars.max():.4f}   Min (active): {active_vars.min():.4f}",
            )

        # ── SAE Diag 4 — Decoded one-hot vs mean active input distance (pixel) ──
        print(
            "\n=== [SparseAE] Diagnostic 4: Decoded one-hot vs mean active input distance ===",  # noqa: E501
        )

        sae_proto_input_dist = np.zeros(N_CONCEPTS)
        sae_proto_input_dist_normalized = np.zeros(N_CONCEPTS)

        for i in range(N_CONCEPTS):
            mask_i = sae_active_mask[:, i].numpy().astype(bool)
            if mask_i.sum() == 0:
                sae_proto_input_dist[i] = np.nan
                sae_proto_input_dist_normalized[i] = np.nan
                continue
            mean_latent_val_i = float(sae_latents[mask_i, i].mean())
            z = torch.zeros(1, N_CONCEPTS, device=device)
            z[0, i] = mean_latent_val_i
            with torch.inference_mode():
                proto_recon_i = sparseae.decode(z).cpu().numpy()[0]  # (input_dim,)
            mean_input_i = X_np[mask_i].mean(axis=0)  # (input_dim,)
            dist = np.linalg.norm(proto_recon_i - mean_input_i)
            sae_proto_input_dist[i] = dist
            sae_proto_input_dist_normalized[i] = dist / (
                np.linalg.norm(proto_recon_i) + 1e-8
            )

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, vals, ylabel, title_suffix in zip(
            axes,
            [sae_proto_input_dist, sae_proto_input_dist_normalized],
            ["||decoded_mean_latent_i - mean_input_i||₂", "distance / ||proto_i||₂"],
            ["absolute", "relative to decoded prototype norm"],
            strict=True,
        ):
            valid = ~np.isnan(vals[sae_order])
            ax.bar(x_s[valid], vals[sae_order][valid], color="coral", alpha=0.8)
            ax.set_xticks(x_s)
            ax.set_xticklabels(
                [f"C{i}\n(n={int(sae_total_act[i])})" for i in sae_order],
                fontsize=8,
            )
            ax.set_ylabel(ylabel)
            ax.set_title(f"Decoded one-hot vs mean active input ({title_suffix})")

        fig.suptitle(f"SparseAE  n_concepts={N_CONCEPTS}", fontsize=13)
        fig.tight_layout()
        fig.savefig(
            out_dir / "sae_diag4_prototype_input_distance.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print("Saved sae_diag4_prototype_input_distance.png")

        valid_mask_sae = ~np.isnan(sae_proto_input_dist)
        print("Decoded one-hot vs mean-active-input distances:")
        print(f"  Mean absolute : {sae_proto_input_dist[valid_mask_sae].mean():.4f}")
        print(
            f"  Mean relative : {sae_proto_input_dist_normalized[valid_mask_sae].mean():.4f}",  # noqa: E501
        )

    print("\n=== Summary ===")
    print("Run the script and interpret:")
    print()
    print(
        "| Diagnostic                   | 'Healthy' signal                   | 'Broken' signal                    |",  # noqa: E501
    )
    print(
        "|------------------------------|------------------------------------|------------------------------------|",
    )
    print(
        "| 1 — alignment dist           | Active clearly right-shifted       | Active ≈ inactive distributions    |",  # noqa: E501
    )
    print(
        "| 2 — prototype vs avg-embed   | Images look similar                | Images look completely different   |",  # noqa: E501
    )
    print(
        "| 3 — within-concept variance  | Low variance                       | High variance                      |",  # noqa: E501
    )
    print(
        "| 4 — prototype-embed distance | Small distance / relative dist ≈ 0 | Large distance / relative dist ≫ 0 |",  # noqa: E501
    )


if __name__ == "__main__":
    main()

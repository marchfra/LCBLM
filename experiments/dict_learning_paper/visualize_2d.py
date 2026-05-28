"""Qualitative 2D visualisation for synthetic sweep outputs.

Usage:
    python experiments/dict_learning_paper/visualize_2d.py <run_dir>

<run_dir> is the timestamped directory produced by sweep.py.

Produces 2d_analysis.png: one subplot per model.  Each panel overlays the
val-data scatter (points coloured by dominant concept) with arrows from the
origin to each alive prototype direction.  Dead prototypes (concepts that never
win the argmax on any val sample) are omitted.

Colors are assigned by concept rank (most-fired = color 0, etc.) so each
color uniquely identifies one concept within a subplot.  The same tab10 palette
is reused across models — colors are NOT comparable across subplots.

For runs with multiple sparsity values the entry with the lowest val MSE is
selected for visualisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_MODEL_ORDER = (
    "vaee",
    "vaee_shared_encoder",
    "topk_sae",
    "sae_concept",
    "vq_vae",
    "beta_vae",
)
_MODEL_LABELS = {
    "vaee": "VAEE",
    "vaee_shared_encoder": "VAEE-SE",
    "topk_sae": "TopK-SAE",
    "sae_concept": "L1-SAE",
    "vq_vae": "VQ-VAE",
    "beta_vae": "β-VAE",
}

_CMAP = plt.get_cmap("tab10")
_N_COLS = 3


def _load_best(run_dir: Path, model_name: str) -> dict | None:
    # New format: model_param=val/results.json subdirectories
    candidates = sorted(run_dir.glob(f"{model_name}_*/results.json"))
    best: dict | None = None
    best_mse = float("inf")
    for p in candidates:
        with p.open() as fh:
            d = json.load(fh)
        mse = d.get("best_val_recon", float("inf"))
        if mse < best_mse:
            best_mse = mse
            best = d
    return best


def _draw_arrow(ax: plt.Axes, tip: np.ndarray, color: object) -> None:
    """Arrow from origin to tip (tip already in display units)."""
    ax.annotate(
        "",
        xy=tip,
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0),
        zorder=5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="2D qualitative analysis plot.")
    parser.add_argument("run_dir", help="Timestamped sweep output directory.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    gt_path = run_dir / "ground_truth.json"
    if not gt_path.exists():
        msg = (
            f"ground_truth.json not found in {run_dir}. "
            "Run sweep.py with a synthetic config first."
        )
        raise FileNotFoundError(msg)

    with gt_path.open() as fh:
        gt = json.load(fh)
    val_data = np.array(gt["val_data_2d"])  # [N, 2]

    n_models = len(_MODEL_ORDER)
    n_cols = _N_COLS
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes_flat = axes.flatten()

    for idx, model_name in enumerate(_MODEL_ORDER):
        ax = axes_flat[idx]
        label = _MODEL_LABELS[model_name]
        ax.set_title(label, fontsize=11, fontweight="bold")

        data = _load_best(run_dir, model_name)
        if data is None:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        dominant = data.get("val_dominant_concept")  # list[int], one per val sample
        prototypes = data.get("prototypes_2d")  # list[list[float]], one per concept

        # ── concept → display colour ──────────────────────────────────────────
        # Rank alive concepts by frequency so the most-fired gets color 0, etc.
        # This avoids duplicate colors and makes the palette assignment stable.
        if dominant is not None:
            from collections import Counter

            counts = Counter(dominant)
            # alive_concepts: concept indices that appear at least once, sorted by
            # frequency descending so color 0 = most common concept
            alive_sorted = [c for c, _ in counts.most_common()]
            concept_color: dict[int, object] = {
                c: _CMAP(i % 10) for i, c in enumerate(alive_sorted)
            }
        else:
            concept_color = {}

        # ── scatter ───────────────────────────────────────────────────────────
        if dominant is not None:
            pt_colors = [concept_color[k] for k in dominant]
        else:
            pt_colors = ["#aaaaaa"] * len(val_data)
        ax.scatter(
            val_data[:, 0], val_data[:, 1], c=pt_colors, s=6, alpha=0.5, linewidths=0
        )

        # ── alive prototype arrows (unit-normalised for visual consistency) ───
        if prototypes is not None and dominant is not None:
            for k, proto in enumerate(prototypes):
                if k not in concept_color:
                    continue  # dead prototype — skip
                tip = np.array(proto, dtype=float)
                norm = np.linalg.norm(tip)
                if norm > 1e-8:
                    tip = tip / norm  # normalise to unit length
                _draw_arrow(ax, tip, color=concept_color[k])

        # ── axis limits ───────────────────────────────────────────────────────
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        lim = max(np.abs(val_data).max(), 1.0) * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#dddddd", lw=0.5, zorder=0)
        ax.axvline(0, color="#dddddd", lw=0.5, zorder=0)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_axis_off()

    fig.suptitle(
        f"{run_dir.name}  —  colour = concept (rank by frequency within model)",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    out_path = run_dir / "2d_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

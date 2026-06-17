"""Preview the 2D complex-synthetic data (per-concept subspaces) for both tiers.

A 2x2 grid:
  rows = easy tier (8 atoms, single active) / medium tier (5 atoms, adjacent pairs)
  cols = make_synthetic (1-D atoms -> round clusters)
         make_complex_synthetic (anchor + tangential 1-D subspace -> strokes)

Same atoms, support and seed within a row, so the only difference is the subspace.

Usage:
    python experiments/dict_learning_paper/plot_complex_data.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import torch

from lcblm.data.synthetic import make_complex_synthetic, make_synthetic

_OUT = Path(__file__).parent / "outputs" / "complex_data_preview.png"
_CMAP = mpl.colormaps["tab10"]
_SUBSPACE_SCALE = 0.3

# Tier parameter sets, matching configs/sweep_synthetic_{easy,medium}.toml.
_TIERS = {
    "easy — 8 atoms, single active": dict(  # noqa: C408
        n_samples=4000,
        n_features=8,
        n_active=1,
        input_dim=2,
        binary_coefs=True,
        noise_std=0.05,
        seed=0,
    ),
    "medium — 5 atoms, adjacent pairs": dict(  # noqa: C408
        n_samples=4000,
        n_features=5,
        input_dim=2,
        active_prob=0.5,
        adjacent_only=True,
        binary_coefs=True,
        noise_std=0.05,
        min_separation=0.628318,
        seed=0,
    ),
}


def _scatter(ax, data: torch.Tensor, atoms: torch.Tensor, title: str) -> None:
    # Colour each point by its nearest anchor (a readable proxy for concept).
    nearest = (data @ atoms.T).argmax(dim=1)
    for i in range(atoms.shape[0]):
        pts = data[nearest == i]
        ax.scatter(
            pts[:, 0], pts[:, 1], s=4, color=_CMAP(i % 10), alpha=0.35, linewidths=0
        )
        ax.annotate(
            "",
            xy=(atoms[i, 0], atoms[i, 1]),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": _CMAP(i % 10), "lw": 1.6},
        )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)


def plot() -> None:
    fig, axes = plt.subplots(len(_TIERS), 2, figsize=(12, 6 * len(_TIERS)))
    for row, (tier, kw) in enumerate(_TIERS.items()):
        ds_simple, atoms = make_synthetic(**kw)
        ds_complex, atoms_c = make_complex_synthetic(
            subspace_rank=1, subspace_scale=_SUBSPACE_SCALE, **kw
        )
        _scatter(
            axes[row][0], ds_simple.data, atoms, f"{tier}\nmake_synthetic (clusters)"
        )
        _scatter(
            axes[row][1],
            ds_complex.data,
            atoms_c,
            f"{tier}\nmake_complex_synthetic (tangential strokes, scale={_SUBSPACE_SCALE})",
        )
    fig.suptitle(
        "2D complex synthetic — each concept becomes a tangential stroke, not a dot",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_OUT}")


if __name__ == "__main__":
    plot()

"""Data preview: medium (noise_std=0.05) vs high (noise_std=0.5) synthetic tiers.

Usage:
    python experiments/dict_learning_paper/plot_high_data.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from lcblm.data.synthetic import make_synthetic

_OUT = Path(__file__).parent / "outputs" / "high_data_preview.png"
_OUT.parent.mkdir(parents=True, exist_ok=True)

_N_SAMPLES_PLOT = 5000
_SEED = 42

# Medium and Hard share the same support recipe — independent Bernoulli (p=0.5)
# activation with the adjacent-only constraint (50% singletons + 50% ring-adjacent
# pairs), sigma=0.05. The difference is the coefficients: medium uses binary
# {0,+1} (tight clusters), hard uses continuous per-sample magnitudes
# c~U(0.5,1.5) (radial smears for singletons, filled patches for pairs).
_CONFIGS = [
    dict(
        label="Medium (binary coefs)",
        active_prob=0.5,
        adjacent_only=True,
        noise_std=0.05,
        coef_range=None,
    ),
    dict(
        label="Hard (continuous coefs c~U(0.5,1.5))",
        active_prob=0.5,
        adjacent_only=True,
        noise_std=0.05,
        coef_range=(0.5, 1.5),
    ),
]

_SHARED = dict(
    n_samples=_N_SAMPLES_PLOT,
    n_features=5,
    input_dim=2,
    binary_coefs=True,
    min_separation=2 * math.pi / 5,
    seed=_SEED,
)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

for ax, cfg in zip(axes, _CONFIGS):
    ds, features = make_synthetic(
        **_SHARED,
        active_prob=cfg["active_prob"],
        adjacent_only=cfg["adjacent_only"],
        noise_std=cfg["noise_std"],
        coef_range=cfg["coef_range"],
    )
    xy = ds.data.numpy()
    feat = features.numpy()

    ax.scatter(xy[:, 0], xy[:, 1], s=3, alpha=0.25, c="#4c72b0")

    scale = 1.3
    for i, f in enumerate(feat):
        ax.annotate(
            "",
            xy=(f[0] * scale, f[1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.8),
        )
        ax.text(
            f[0] * scale * 1.08,
            f[1] * scale * 1.08,
            str(i),
            fontsize=8,
            color="crimson",
        )

    r = 1.8
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect("equal")
    ax.axhline(0, lw=0.4, color="gray")
    ax.axvline(0, lw=0.4, color="gray")
    ax.set_title(cfg["label"], fontsize=11)

fig.suptitle(
    "Synthetic 2D: 5 atoms, adjacent-only support — binary (medium) vs continuous coefs (hard)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(_OUT, dpi=150, bbox_inches="tight")
print(f"Saved to {_OUT}")

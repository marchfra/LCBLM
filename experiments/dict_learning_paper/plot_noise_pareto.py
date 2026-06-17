"""MSE-vs-#active Pareto across within-concept-variation (noise) levels.

The high-dim hard tier was re-run at three additive-noise levels (σ = 0.05,
0.10, 0.20) with dead-latent resampling on, to test whether VAEE's
reconstruction-per-active-concept lead widens as within-concept variation grows.
This plots every swept config as a point (MSE vs #active concepts), one panel
per noise level, coloured by model, annotated with the recovery (matched)
fraction — the honest multi-axis view (reconstruction, sparsity, recovery).

Usage:
    python experiments/dict_learning_paper/plot_noise_pareto.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).parent
_OUT = _ROOT / "outputs" / "noise_pareto.png"

_LEVELS = [("005", 0.05), ("010", 0.10), ("020", 0.20)]
_MODELS = {
    "vaee": ("VAEE", "#d62728", "o"),
    "vaee_shared_encoder": ("VAEE-SE", "#ff9896", "s"),
    "topk_sae": ("TopK-SAE", "#1f77b4", "^"),
    "sae_concept": ("L1-SAE", "#2ca02c", "D"),
}


def _points(run_dir: Path) -> dict[str, list[tuple[float, float, float, str]]]:
    """{model: [(mse, n_active, matched, cfg), ...]} for every config."""
    pts: dict[str, list] = {m: [] for m in _MODELS}
    for f in glob.glob(str(run_dir / "*/results.json")):
        d = json.loads(Path(f).read_text())
        m = d["model_name"]
        if m not in pts or not d.get("val_recon"):
            continue
        bi = int(np.argmin(d["val_recon"]))
        mse = d["val_recon"][bi]
        active = d["val_l0"][bi] if d.get("val_l0") else float("nan")
        matched = d.get("matched_fraction", float("nan"))
        cfg = Path(f).parent.name[len(m) + 1 :]
        pts[m].append((mse, active, matched, cfg))
    return pts


def plot() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    fig.suptitle(
        "Within-concept variation: MSE vs #active concepts (point label = recovery / matched fraction)\n"
        "high-dim hard tier, D=32, K=64 atoms, E[k]=3, dead-latent resampling on",
        fontsize=12,
    )
    for ax, (tag, sigma) in zip(axes, _LEVELS, strict=True):
        run_dir = sorted(
            Path(_ROOT / "outputs").glob(
                f"sweep_synthetic_highdim_hard_noise{tag}/*/"
            )
        )[-1]
        pts = _points(run_dir)
        for m, (label, color, marker) in _MODELS.items():
            for mse, active, matched, _cfg in pts[m]:
                ax.scatter(
                    active,
                    mse,
                    s=70,
                    c=color,
                    marker=marker,
                    edgecolors="k",
                    linewidths=0.4,
                    zorder=3,
                )
                ax.annotate(
                    f"{matched:.2f}",
                    (active, mse),
                    fontsize=6.5,
                    xytext=(3, 3),
                    textcoords="offset points",
                    color=color,
                )
        ax.set_yscale("log")
        ax.set_title(f"σ = {sigma}", fontsize=11)
        ax.set_xlabel("# active concepts")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("val recon MSE (log)")
    handles = [
        plt.Line2D(
            [], [], color=c, marker=mk, linestyle="", markeredgecolor="k", label=lab
        )
        for lab, c, mk in _MODELS.values()
    ]
    axes[-1].legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_OUT}")


if __name__ == "__main__":
    plot()

"""Embedding-size Pareto (MSE vs #active concepts), high-dim hard, sigma=0.05.

Plots the VAEE *and* VAEE-SE embedding-size trajectories (E = 4, 8, 16, 32, 64,
128) against the TopK-SAE and L1-SAE baselines on the same tier. Every point is
labelled with its recovery (matched) fraction — the honest third axis.

Point sources (all sigma=0.05, pi=0.047, resample on):
  * E=4         -> noise005 run dir (vaee / vaee_shared_encoder, pi=0.047)
  * E=8,16,32   -> embsweep (VAEE) and embsweep_se (VAEE-SE)
  * E=64,128    -> embsweep_hi (both variants)
  * TopK / L1   -> noise005 run dir

Usage:
    python experiments/dict_learning_paper/plot_emb_pareto.py
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
_OUT = _ROOT / "outputs" / "emb_pareto.png"


def _latest(pattern: str) -> Path | None:
    dirs = sorted((_ROOT / "outputs").glob(pattern))
    return dirs[-1] if dirs else None


_NOISE005 = _latest("sweep_synthetic_highdim_hard_noise005/*/")
_EMB = _latest("sweep_synthetic_highdim_hard_embsweep/*/")
_EMB_SE = _latest("sweep_synthetic_highdim_hard_embsweep_se/*/")
_EMB_HI = _latest("sweep_synthetic_highdim_hard_embsweep_hi/*/")


def _load(run_dir: Path | None) -> dict | None:
    if run_dir is None or not (run_dir / "results.json").exists():
        return None
    r = json.loads((run_dir / "results.json").read_text())
    if not r.get("val_recon"):
        return None
    bi = int(np.argmin(r["val_recon"]))
    return {
        "mse": r["val_recon"][bi],
        "active": r["val_l0"][bi],
        "matched": r.get("matched_fraction", float("nan")),
    }


def _trajectory(variant: str) -> list[dict]:
    """variant in {'vaee', 'vaee_shared_encoder'}; one point per E."""
    sources = {
        4: (_NOISE005, f"{variant}_pi=0.047"),
        8: ((_EMB if variant == "vaee" else _EMB_SE), f"{variant}_embedding_size=8"),
        16: ((_EMB if variant == "vaee" else _EMB_SE), f"{variant}_embedding_size=16"),
        32: ((_EMB if variant == "vaee" else _EMB_SE), f"{variant}_embedding_size=32"),
        64: (_EMB_HI, f"{variant}_embedding_size=64"),
        128: (_EMB_HI, f"{variant}_embedding_size=128"),
    }
    pts = []
    for e, (base, sub) in sources.items():
        d = _load(base / sub) if base else None
        if d:
            d["E"] = e
            pts.append(d)
    return sorted(pts, key=lambda d: d["E"])


def _baseline(model: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(_NOISE005 / f"{model}_*/results.json"))):
        d = _load(Path(f).parent)
        if d:
            d["cfg"] = Path(f).parent.name.split("_")[-1]
            out.append(d)
    return out


def _plot_traj(ax, pts: list[dict], color: str, label: str) -> None:
    ax.plot(
        [p["active"] for p in pts],
        [p["mse"] for p in pts],
        "-",
        color=color,
        lw=1.4,
        alpha=0.7,
        zorder=2,
    )
    for p in pts:
        ax.scatter(
            p["active"],
            p["mse"],
            s=55 + 16 * np.log2(p["E"]),
            c=color,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            f"E={p['E']} ({p['matched']:.2f})",
            (p["active"], p["mse"]),
            fontsize=6.8,
            xytext=(5, 3),
            textcoords="offset points",
            color=color,
        )
    ax.annotate(
        label,
        (pts[-1]["active"], pts[-1]["mse"]),
        fontsize=8,
        fontweight="bold",
        xytext=(5, -12),
        textcoords="offset points",
        color=color,
    )


def plot() -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    _plot_traj(ax, _trajectory("vaee"), "#d62728", "VAEE")
    _plot_traj(ax, _trajectory("vaee_shared_encoder"), "#9467bd", "VAEE-SE")

    for model, color, marker, lab in (
        ("topk_sae", "#1f77b4", "^", "TopK-SAE"),
        ("sae_concept", "#2ca02c", "D", "L1-SAE"),
    ):
        for d in _baseline(model):
            ax.scatter(
                d["active"], d["mse"], s=70, c=color, marker=marker,
                edgecolors="k", linewidths=0.4, zorder=3, label=lab,
            )
            ax.annotate(
                f"{lab.split('-')[0]} {d['cfg']} ({d['matched']:.2f})",
                (d["active"], d["mse"]),
                fontsize=6.5,
                xytext=(5, -9),
                textcoords="offset points",
                color=color,
            )

    ax.set_yscale("log")
    ax.set_xlabel("# active concepts (per sample)")
    ax.set_ylabel("val recon MSE (log)")
    ax.set_title(
        "Embedding-size Pareto — high-dim hard, σ=0.05\n"
        "VAEE & VAEE-SE trajectories E=4→128 (label = matched recovery) vs TopK / L1",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([], [], color="#d62728", marker="o", ls="-", label="VAEE (E sweep)"),
        plt.Line2D([], [], color="#9467bd", marker="o", ls="-", label="VAEE-SE (E sweep)"),
        plt.Line2D([], [], color="#1f77b4", marker="^", ls="", label="TopK-SAE"),
        plt.Line2D([], [], color="#2ca02c", marker="D", ls="", label="L1-SAE"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_OUT}")


if __name__ == "__main__":
    plot()

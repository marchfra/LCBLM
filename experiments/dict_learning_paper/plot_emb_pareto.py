"""Pareto (MSE vs #active concepts) for the embedding-size sweep, sigma=0.05.

Shows the VAEE embedding-size trajectory (E = 4 -> 8 -> 16 -> 32) against the
TopK-SAE and L1-SAE baselines on the same high-dim hard tier (sigma=0.05).
E=4 is the noise005 pi=0.047 reference; E in {8,16,32} come from the dedicated
embedding-size sweep. Every point is labelled with its recovery (matched)
fraction — the honest third axis.

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
_NOISE005 = sorted(
    (_ROOT / "outputs").glob("sweep_synthetic_highdim_hard_noise005/*/")
)[-1]
_EMB = sorted((_ROOT / "outputs").glob("sweep_synthetic_highdim_hard_embsweep/*/"))[-1]


def _load(path: Path) -> dict | None:
    p = path / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    if not r.get("val_recon"):
        return None
    bi = int(np.argmin(r["val_recon"]))
    return {
        "mse": r["val_recon"][bi],
        "active": r["val_l0"][bi],
        "matched": r.get("matched_fraction", float("nan")),
    }


def _baseline(model: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(_NOISE005 / f"{model}_*/results.json"))):
        d = _load(Path(f).parent)
        if d:
            d["cfg"] = Path(f).parent.name.split("_")[-1]
            out.append(d)
    return out


def plot() -> None:
    # VAEE embedding-size trajectory: E=4 from noise005 (pi=0.047), 8/16/32 from embsweep.
    emb = []
    e4 = _load(_NOISE005 / "vaee_pi=0.047")
    if e4:
        e4["E"] = 4
        emb.append(e4)
    for E in (8, 16, 32):
        d = _load(_EMB / f"vaee_embedding_size={E}")
        if d:
            d["E"] = E
            emb.append(d)
    emb.sort(key=lambda d: d["E"])

    fig, ax = plt.subplots(figsize=(8.2, 6))

    # VAEE trajectory (connected, marker size grows with E).
    xs = [d["active"] for d in emb]
    ys = [d["mse"] for d in emb]
    ax.plot(xs, ys, "-", color="#d62728", lw=1.4, zorder=2, alpha=0.7)
    # Staggered label offsets so the crowded E=4..32 cluster stays readable.
    _off = {4: (10, 14), 8: (10, -2), 16: (12, -16), 32: (10, -28)}
    for d in emb:
        ax.scatter(
            d["active"],
            d["mse"],
            s=60 + 18 * np.log2(d["E"]),
            c="#d62728",
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            f"VAEE E={d['E']}  (matched {d['matched']:.2f})",
            (d["active"], d["mse"]),
            fontsize=7.5,
            xytext=_off.get(d["E"], (6, 4)),
            textcoords="offset points",
            color="#7a1416",
        )

    # SAE baselines.
    for model, color, marker, lab in (
        ("topk_sae", "#1f77b4", "^", "TopK-SAE"),
        ("sae_concept", "#2ca02c", "D", "L1-SAE"),
    ):
        for d in _baseline(model):
            ax.scatter(
                d["active"],
                d["mse"],
                s=70,
                c=color,
                marker=marker,
                edgecolors="k",
                linewidths=0.4,
                zorder=3,
                label=lab,
            )
            ax.annotate(
                f"{lab.split('-')[0]} {d['cfg']}\nmatched {d['matched']:.2f}",
                (d["active"], d["mse"]),
                fontsize=7,
                xytext=(6, -10),
                textcoords="offset points",
                color=color,
            )

    ax.set_yscale("log")
    ax.set_xlabel("# active concepts (per sample)")
    ax.set_ylabel("val recon MSE (log)")
    ax.set_title(
        "High-dim hard, σ=0.05 — embedding-size Pareto\n"
        "VAEE E=4→32 trajectory (red) vs TopK / L1; label = recovery (matched)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    # De-duplicate legend labels.
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h, strict=True))
    red = plt.Line2D([], [], color="#d62728", marker="o", linestyle="-", label="VAEE (E sweep)")
    ax.legend(
        [red, *seen.values()], ["VAEE (E sweep)", *seen.keys()], fontsize=8, loc="upper right"
    )
    fig.tight_layout()
    fig.savefig(_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_OUT}")


if __name__ == "__main__":
    plot()

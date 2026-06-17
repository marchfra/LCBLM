"""PCA-projected qualitative plot for high-dimensional synthetic sweeps.

visualize_2d.py only works when input_dim==2. For high-dim tiers this script
projects the ground-truth atoms, each model's learned prototypes, and the
validation data onto a shared 2-D PCA basis (fit on the atoms) so prototype↔atom
alignment can be eyeballed. Note the projection is lossy — with K atoms in D≫2
dims many atoms overlap in 2-D — so read it qualitatively (do prototypes cluster
on atoms, or scatter?), with the matched/cos numbers in each panel title.

Usage:
    python experiments/dict_learning_paper/visualize_pca.py <run_dir> [--pin model=substr]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
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
_N_COLS = 3


def _pca_axes(atoms: np.ndarray) -> np.ndarray:
    """Top-2 right singular vectors of the (uncentered) atom matrix → [2, D].

    Uncentered so the origin maps to 0 and direction vectors stay comparable.
    """
    _u, _s, vt = np.linalg.svd(atoms, full_matrices=False)
    return vt[:2]


def _load_best(
    run_dir: Path, model: str, pin: dict[str, str]
) -> tuple[dict, str] | None:
    longer = [m + "_" for m in _MODEL_ORDER if m != model and m.startswith(model + "_")]
    pin_sub = pin.get(model)
    cands = [
        p
        for p in sorted(run_dir.glob(f"{model}_*/results.json"))
        if not any(p.parent.name.startswith(pre) for pre in longer)
        and (pin_sub is None or pin_sub in p.parent.name)
    ]
    best = best_label = best_key = None
    for p in cands:
        d = json.loads(p.read_text())
        matched = d.get("matched_fraction")
        key = (
            matched if matched is not None else -1.0,
            d.get("mean_cosine_sim") or 0.0,
        )
        if best_key is None or key > best_key:
            best, best_label, best_key = d, p.parent.name[len(model) + 1 :], key
    return (best, best_label) if best is not None else None


def plot(run_dir: Path, pin: dict[str, str]) -> None:
    gt = json.loads((run_dir / "ground_truth.json").read_text())
    atoms = np.asarray(gt["atoms_2d"], dtype=float)  # [K, D] (high-D despite name)
    val = np.asarray(gt["val_data_2d"], dtype=float)  # [N, D]
    axes = _pca_axes(atoms)  # [2, D] shared basis
    atoms_2d = atoms @ axes.T
    val_2d = (val @ axes.T)[: min(len(val), 2000)]

    n_rows = (len(_MODEL_ORDER) + _N_COLS - 1) // _N_COLS
    fig, axs = plt.subplots(n_rows, _N_COLS, figsize=(4.5 * _N_COLS, 4.5 * n_rows))
    axs = axs.flatten()
    var = float((atoms @ axes.T).var(0).sum() / (atoms.var(0).sum() + 1e-12))
    fig.suptitle(
        f"{run_dir.name} — PCA-2D projection (atoms basis; ~{var:.0%} atom variance shown)",
        fontsize=12,
    )

    for idx, model in enumerate(_MODEL_ORDER):
        ax = axs[idx]
        ax.scatter(val_2d[:, 0], val_2d[:, 1], s=4, alpha=0.12, c="#999", zorder=1)
        ax.scatter(
            atoms_2d[:, 0],
            atoms_2d[:, 1],
            marker="x",
            s=40,
            c="k",
            linewidths=1.2,
            label="GT atoms",
            zorder=3,
        )
        loaded = _load_best(run_dir, model, pin)
        if loaded is None:
            ax.set_title(_MODEL_LABELS[model], fontsize=11, fontweight="bold")
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            continue
        d, lbl = loaded
        protos = np.asarray(d.get("prototypes_2d", []), dtype=float)
        if protos.size:
            p2d = protos @ axes.T
            ax.scatter(
                p2d[:, 0],
                p2d[:, 1],
                s=28,
                c="crimson",
                alpha=0.8,
                label="learned",
                zorder=4,
            )
        m, c = d.get("matched_fraction"), d.get("mean_cosine_sim")
        title = f"{_MODEL_LABELS[model]} ({lbl})"
        sub = f"matched={m:.2f}  cos={c:.2f}" if m is not None else ""
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(0.02, 0.98, sub, transform=ax.transAxes, fontsize=8, va="top")
        ax.set_aspect("equal")
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    for ax in axs[len(_MODEL_ORDER) :]:
        ax.set_axis_off()
    fig.tight_layout()
    out = run_dir / "pca_analysis.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--pin", action="append", default=[], metavar="MODEL=SUBSTR")
    args = p.parse_args()
    plot(Path(args.run_dir), dict(x.split("=", 1) for x in args.pin))


if __name__ == "__main__":
    main()

"""Qualitative 2D visualisation for synthetic sweep outputs.

Usage:
    python experiments/dict_learning_paper/visualize_2d.py <run_dir>

<run_dir> is the timestamped directory produced by sweep.py.

Produces 2d_analysis.png: one subplot per model.  Each panel overlays the
val-data scatter with arrows from the origin to each *alive* prototype direction.
Each point is coloured as the alpha-weighted sum of its active concepts' colours,
so points that mix several concepts blend toward a mixture colour (falling back to
argmax-dominant colouring for runs saved before val_concept_weights).  A concept is
alive if it fires on >=0.1% of val samples (the same criterion as alive_dict_size),
so a concept can have an arrow without ever winning the per-sample argmax — this is
common for VAEE's independent Bernoulli gates.  Concepts that never fire are
omitted.

Colors are assigned by concept rank (highest total alpha mass = color 0, etc.) so
each color uniquely identifies one concept within a subplot.  The same tab10 palette
is reused across models — colors are NOT comparable across subplots.

For runs with multiple sparsity values the entry with the lowest total validation
loss is selected — the same criterion used by training early stopping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
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


def _selection_score(d: dict) -> tuple[float, float, float]:
    """Rank key for picking which sweep run to visualise (higher is better).

    Prefers the best atom recovery when available — highest ``matched_fraction``,
    tie-broken by ``mean_cosine_sim``, then by lowest ``best_val_total`` (negated).
    Falls back to lowest ``best_val_total`` alone (then ``best_val_recon``) when no
    recovery metrics are saved. The recovery-first ordering keeps the picking sane
    when sweeping a regularisation weight directly (e.g. ``gamma`` for VAEE),
    where the unweighted ``best_val_total`` trivially favours the smallest weight.
    """
    matched = d.get("matched_fraction")
    cos = d.get("mean_cosine_sim", 0.0) or 0.0
    total = d.get("best_val_total")
    if total is None:
        total = d.get("best_val_recon", float("inf"))
    if matched is None:
        return (-1.0, 0.0, -total)
    return (matched, cos, -total)


def _load_best(
    run_dir: Path, model_name: str, pin: dict[str, str] | None = None
) -> tuple[dict, str] | None:
    """Return *(results, param_label)* for the best-scoring run of *model_name*.

    ``param_label`` is the swept config the run dir encodes (the ``param=value``
    suffix of ``{model_name}_{param}={value}``), shown beside the model name in the
    subplot title. Returns ``None`` when the model has no runs in *run_dir*.

    ``pin`` maps a model name to a substring; only runs whose label contains it are
    considered (e.g. ``{"vq_vae": "num_codes=5"}`` forces the fair codebook instead
    of the best-matched degenerate one).
    """
    pin_sub = (pin or {}).get(model_name)
    # New format: model_param=val/results.json subdirectories.
    # Exclude dirs of a longer model whose name has this one as a prefix
    # (e.g. "vaee" must not match "vaee_shared_encoder_*").
    longer = [
        m + "_"
        for m in _MODEL_ORDER
        if m != model_name and m.startswith(model_name + "_")
    ]
    candidates = [
        p
        for p in sorted(run_dir.glob(f"{model_name}_*/results.json"))
        if not any(p.parent.name.startswith(pre) for pre in longer)
        and (pin_sub is None or pin_sub in p.parent.name)
    ]
    best: dict | None = None
    best_label = ""
    best_key: float | None = None
    for p in candidates:
        with p.open() as fh:
            d = json.load(fh)
        key = _selection_score(d)
        if best_key is None or key > best_key:
            best_key = key
            best = d
            # Strip the "{model_name}_" prefix to leave the swept "param=value".
            best_label = p.parent.name[len(model_name) + 1 :]
    if best is None:
        return None
    return best, best_label


def _blend_colors(
    weights: list[list[float]] | None,
    concept_color: dict[int, object],
    n_points: int,
    dominant: list[int] | None,
) -> list:
    """Per-point RGBA from the alpha-weighted average of active concepts' colours.

    ``weights`` is the ``[N, K]`` activation matrix. Only concepts present in
    ``concept_color`` contribute (dead concepts carry zero weight); each point's
    colour is ``sum_k w_k * rgb_k / sum_k w_k``. Points whose coloured concepts all
    have zero weight render grey. Falls back to argmax-dominant colouring (then grey)
    when ``weights`` is unavailable.
    """
    if weights is None:
        if dominant is not None:
            return [concept_color[k] for k in dominant]
        return ["#aaaaaa"] * n_points

    w = np.asarray(weights, dtype=float)  # [N, K]
    idx = sorted(concept_color)
    palette = np.array([mcolors.to_rgb(concept_color[k]) for k in idx])  # [C, 3]
    sub = w[:, idx] if idx else np.zeros((w.shape[0], 0))  # [N, C]
    totals = sub.sum(axis=1, keepdims=True)  # [N, 1]
    blended = np.where(
        totals > 0, (sub @ palette) / np.where(totals > 0, totals, 1.0), np.nan
    )
    grey = np.array(mcolors.to_rgb("#aaaaaa"))
    rgb = np.where(np.isnan(blended), grey, blended)
    return [tuple(c) for c in rgb]


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
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="MODEL=SUBSTR",
        help="Force a model's panel to a run whose label contains SUBSTR "
        "(e.g. --pin vq_vae=num_codes=5). Repeatable.",
    )
    args = parser.parse_args()
    pin = dict(p.split("=", 1) for p in args.pin)

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

        best = _load_best(run_dir, model_name, pin)
        if best is None:
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        data, param_label = best
        title = f"{label}  ({param_label})" if param_label else label
        ax.set_title(title, fontsize=11, fontweight="bold")

        dominant = data.get("val_dominant_concept")  # list[int], one per val sample
        weights = data.get("val_concept_weights")  # list[list[float]] — [N, K] alpha
        prototypes = data.get("prototypes_2d")  # list[list[float]], one per concept
        alive = data.get("alive_concepts")  # list[int] — fire on >=0.1% of val samples

        # ── concept → display colour ──────────────────────────────────────────
        # Rank concepts by total alpha mass (sum of val_concept_weights) so colour 0
        # = the concept carrying the most activation. Colours are restricted to alive
        # concepts (which get arrows) plus any argmax winner, keeping the palette
        # bounded and meaningful. Falls back to argmax-frequency ranking for runs
        # saved before val_concept_weights existed.
        concept_color: dict[int, object] = {}
        relevant: list[int] = list(alive) if alive is not None else []
        if dominant is not None:
            relevant += [c for c in dict.fromkeys(dominant) if c not in relevant]

        if weights is not None and relevant:
            mass = np.asarray(weights, dtype=float).sum(axis=0)  # [K]
            ranked = sorted(
                relevant,
                key=lambda c: mass[c] if c < len(mass) else 0.0,
                reverse=True,
            )
            concept_color = {c: _CMAP(i % 10) for i, c in enumerate(ranked)}
        elif dominant is not None:
            from collections import Counter

            counts = Counter(dominant)
            ranked = [c for c, _ in counts.most_common()]
            if alive is not None:
                ranked += [c for c in alive if c not in counts]
            concept_color = {c: _CMAP(i % 10) for i, c in enumerate(ranked)}

        # ── scatter (points coloured by alpha-weighted concept blend) ─────────
        # Each point's colour is the per-sample, alpha-weighted average of its
        # active concepts' RGB colours. Concepts without a colour (dead) carry zero
        # weight. Samples with no active coloured concept fall back to grey. Older
        # runs without val_concept_weights fall back to argmax-dominant colouring.
        pt_colors = _blend_colors(weights, concept_color, len(val_data), dominant)
        ax.scatter(
            val_data[:, 0], val_data[:, 1], c=pt_colors, s=6, alpha=0.5, linewidths=0
        )

        # ── alive prototype arrows (unit-normalised for visual consistency) ───
        # One arrow per *alive* concept (matches alive_dict_size), not just argmax
        # winners. Falls back to the dominant set for runs saved before alive_concepts.
        if alive is not None:
            arrow_concepts = alive
        elif dominant is not None:
            arrow_concepts = list(dict.fromkeys(dominant))
        else:
            arrow_concepts = []
        if prototypes is not None:
            for k in arrow_concepts:
                if k >= len(prototypes):
                    continue
                tip = np.array(prototypes[k], dtype=float)
                norm = np.linalg.norm(tip)
                if norm > 1e-8:
                    tip = tip / norm  # normalise to unit length
                _draw_arrow(ax, tip, color=concept_color.get(k, "#888888"))

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
        f"{run_dir.name}  —  colour = alpha-weighted concept blend (rank by mass)",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    out_path = run_dir / "2d_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

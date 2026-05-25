"""L0-vs-MSE scatter plot CLI.

Reads results.json and config.json from run directories produced by ct-train.
Points are grouped by their second hyperparameter (latent_dim for TopK SAE,
embedding_size for VAEE); points within a group are connected by a line.
Colour encodes the group; marker shape encodes model type.

Usage
-----
    ct-plot outputs/VAEE-50x64-* outputs/TopK-64-SAE-4096-*
    ct-plot outputs/  # processes all immediate subdirs that contain results.json
    ct-plot outputs/VAEE-50x64-* --csv metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter

from lcblm.utils.plotting import set_plt_style

mpl.use("Agg")

_TIMESTAMP_RE = r"-\d{8}_\d{6}$"

# Colour palettes for each model family (light → dark)
_VAEE_PALETTE = ["#90CAF9", "#1E88E5", "#0D47A1"]  # blues
_SAE_PALETTE = ["#FFCC80", "#F57C00", "#BF360C"]  # oranges

_MODEL_MARKERS = {"vaee": "^", "topk_sae": "o"}
_MODEL_LABELS = {"vaee": "VAEE", "topk_sae": "TopK SAE"}
_DEFAULT_COLOR = "#888888"


def _strip_timestamp(name: str) -> str:
    return re.sub(_TIMESTAMP_RE, "", name)


def _collect_rows(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for d in run_dirs:
        results_path = d / "results.json"
        config_path = d / "config.json"
        if not results_path.exists():
            continue
        with results_path.open() as f:
            r = json.load(f)

        model_name = r.get("model_name", "")
        group_key = None
        group_label = ""
        point_label = ""

        if config_path.exists():
            with config_path.open() as f:
                cfg = json.load(f)
            rc = cfg.get("run_config", {})
            if model_name == "vaee":
                group_key = rc.get("embedding_size")
                group_label = f"emb_size={group_key}"
                point_label = f"N={rc.get('num_embeddings')}"
            elif model_name == "topk_sae":
                group_key = r.get("n_concepts")  # resolved latent_dim
                group_label = f"latent_dim={group_key}"
                point_label = f"k={rc.get('k')}"

        if not group_label:
            group_label = _strip_timestamp(d.name)

        rows.append(
            {
                "name": r.get("run_name") or _strip_timestamp(d.name),
                "model_name": model_name,
                "l0": r["best_l0"],
                "mse": r["best_val_recon"],
                "group_key": group_key,
                "group_label": group_label,
                "point_label": point_label,
            },
        )
    return rows


def _build_color_map(rows: list[dict]) -> dict[tuple, str]:
    """Assign a colour to each (model_name, group_key) pair."""
    vaee_keys = sorted(
        {
            r["group_key"]
            for r in rows
            if r["model_name"] == "vaee" and r["group_key"] is not None
        },
    )
    sae_keys = sorted(
        {
            r["group_key"]
            for r in rows
            if r["model_name"] == "topk_sae" and r["group_key"] is not None
        },
    )
    color_map: dict[tuple, str] = {}
    for i, k in enumerate(vaee_keys):
        color_map[("vaee", k)] = _VAEE_PALETTE[i % len(_VAEE_PALETTE)]
    for i, k in enumerate(sae_keys):
        color_map[("topk_sae", k)] = _SAE_PALETTE[i % len(_SAE_PALETTE)]
    return color_map


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser(
        prog="ct-plot",
        description="Generate L0-vs-MSE scatter from ct-train run directories.",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run output dirs (or a parent dir whose subdirs contain results.json).",
    )
    parser.add_argument(
        "--csv",
        default="",
        metavar="PATH",
        help="Also write metrics to a CSV file at this path.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="",
        help="Output PNG path (default: l0_vs_mse.png next to the first run dir).",
    )
    parser.add_argument(
        "--mplstyles",
        default="mplstyles",
        help="Path to matplotlib stylesheet folder (default: mplstyles).",
    )
    args = parser.parse_args()

    # Collect run directories — each arg may be a direct run dir or a parent.
    run_dirs: list[Path] = []
    for raw in args.run_dirs:
        p = Path(raw)
        if (p / "results.json").exists():
            run_dirs.append(p)
        else:
            run_dirs.extend(
                sorted(c for c in p.iterdir() if (c / "results.json").exists()),
            )

    if not run_dirs:
        print("No run directories with results.json found.")
        return

    rows = _collect_rows(run_dirs)
    if not rows:
        print("No results.json files found.")
        return

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "model_name",
                    "l0",
                    "mse",
                    "group_key",
                    "group_label",
                    "point_label",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {csv_path}")

    color_map = _build_color_map(rows)

    # Group rows by (model_name, group_key) for line drawing.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["model_name"], row["group_key"])].append(row)

    set_plt_style(["grid", "science", "notebook", "mylegend"], args.mplstyles)
    fig, ax = plt.subplots(figsize=(10, 6))

    texts: list = []
    all_xs: list[float] = []
    all_ys: list[float] = []

    for (model_name, group_key), group_rows in sorted(groups.items()):
        color = color_map.get((model_name, group_key), _DEFAULT_COLOR)
        marker = _MODEL_MARKERS.get(model_name, "o")

        sorted_rows = sorted(group_rows, key=lambda r: r["l0"])
        xs = [r["l0"] for r in sorted_rows]
        ys = [r["mse"] for r in sorted_rows]

        ax.plot(xs, ys, color=color, lw=1.2, alpha=0.7, zorder=3)
        ax.scatter(
            xs,
            ys,
            color=color,
            marker=marker,
            s=80,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )
        for row in sorted_rows:
            all_xs.append(row["l0"])
            all_ys.append(row["mse"])
            if row["point_label"]:
                t = ax.text(
                    row["l0"],
                    row["mse"],
                    row["point_label"],
                    fontsize=10,
                    ha="left",
                    va="bottom",
                )
                texts.append(t)

    # ── Legend ────────────────────────────────────────────────────────────────
    present_models = {r["model_name"] for r in rows}
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=_MODEL_MARKERS[mn],
            color="0.3",
            lw=0,
            markersize=8,
            label=_MODEL_LABELS[mn],
        )
        for mn in ("vaee", "topk_sae")
        if mn in present_models
    ]

    vaee_keys = sorted(
        {
            r["group_key"]
            for r in rows
            if r["model_name"] == "vaee" and r["group_key"] is not None
        },
    )
    sae_keys = sorted(
        {
            r["group_key"]
            for r in rows
            if r["model_name"] == "topk_sae" and r["group_key"] is not None
        },
    )
    blank = Line2D([0], [0], color="none", label=" ")
    vaee_handles = [
        Line2D([0], [0], color=color_map[("vaee", k)], lw=2, label=f"emb_size={k}")
        for k in vaee_keys
    ]
    sae_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[("topk_sae", k)],
            lw=2,
            label=f"latent_dim={k}",
        )
        for k in sae_keys
    ]

    legend_handles = list(marker_handles)
    if vaee_handles:
        legend_handles += [blank, *vaee_handles]
    if sae_handles:
        legend_handles += [blank, *sae_handles]

    ax.legend(handles=legend_handles, fontsize=10)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([7, 10, 20, 30, 50, 100]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\text{L}_0$")
    ax.set_ylabel(r"Validation $\text{MSE}$")
    # ax.set_title("Sparsity vs Reconstruction Quality", pad=12)
    # ax.grid(lw=0.7, alpha=0.4, which="both")
    ax.grid(axis="x", which="minor", alpha=0.3)

    if texts:
        adjust_text(
            texts,
            x=all_xs,
            y=all_ys,
            ax=ax,
            expand=(1.3, 1.5),
            force_points=(0.4, 0.8),
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
            min_arrow_len=15,
        )
    fig.tight_layout()

    out_path = (
        Path(args.out) if args.out else Path(args.run_dirs[0]).parent / "l0_vs_mse.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

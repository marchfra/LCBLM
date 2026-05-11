"""L0-vs-MSE scatter plot CLI.

Reads results.json from one or more run directories produced by ct-train,
extracts best_l0 and best_val_recon, and produces a scatter plot coloured by
model type. Optionally writes a CSV of the raw metrics.

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
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from adjustText import adjust_text

from lcblm.utils.plotting import set_plt_style

mpl.use("Agg")

_MODEL_COLORS = {
    "vaee": "#2196F3",
    "topk_sae": "#FF6F00",
    "sae_concept": "#4CAF50",
    "sae_param": "#9C27B0",
}
_DEFAULT_COLOR = "#888888"

_TIMESTAMP_RE = r"-\d{8}_\d{6}$"


def _strip_timestamp(name: str) -> str:
    return re.sub(_TIMESTAMP_RE, "", name)


def _collect_rows(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for d in run_dirs:
        results_path = d / "results.json"
        if not results_path.exists():
            continue
        with results_path.open() as f:
            r = json.load(f)
        run_name = r.get("run_name") or _strip_timestamp(d.name)
        rows.append(
            {
                "name": run_name,
                "model_name": r.get("model_name", ""),
                "l0": r["best_l0"],
                "mse": r["best_val_recon"],
            },
        )
    return rows


def main() -> None:  # noqa: PLR0915
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
        "--max-l0",
        type=int,
        default=30,
        dest="max_l0",
        help="Clip the x-axis at this L0 value (default: 30).",
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
            writer = csv.DictWriter(f, fieldnames=["name", "model_name", "l0", "mse"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to {csv_path}")

    x_max = args.max_l0

    set_plt_style(["grid", "science", "notebook", "mylegend"], args.mplstyles)
    fig, ax = plt.subplots(figsize=(10, 6))

    seen_models: set[str] = set()
    auto_texts = []
    all_xs: list[float] = []
    all_ys: list[float] = []

    for row in rows:
        model_name = row["model_name"]
        color = _MODEL_COLORS.get(model_name, _DEFAULT_COLOR)
        x = min(row["l0"], x_max)
        y = row["mse"]
        clipped = row["l0"] > x_max
        all_xs.append(x)
        all_ys.append(y)

        label = model_name if model_name not in seen_models else None
        seen_models.add(model_name)
        ax.scatter(
            x,
            y,
            color=color,
            s=90,
            zorder=4,
            marker=">" if clipped else "o",
            label=label,
        )

        display = row["name"].replace("-", " ")
        if clipped:
            ax.annotate(
                f"{display}\n(L0 = {row['l0']:.0f})",
                (x_max, y),
                textcoords="offset points",
                xytext=(-8, 3),
                color=color,
                ha="right",
                fontsize=9,
            )
        else:
            t = ax.text(x, y, display, color=color, fontsize=9, ha="left")
            auto_texts.append(t)

    if auto_texts:
        adjust_text(
            auto_texts,
            x=all_xs,
            y=all_ys,
            ax=ax,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
        )

    ax.set_xlim(right=x_max)
    ax.set_xlabel("L0 (mean active concepts per token)")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Sparsity vs Reconstruction Quality", pad=12)
    ax.legend(fontsize=12)
    ax.grid(lw=0.7, alpha=0.4)
    fig.tight_layout()

    out_path = (
        Path(args.out) if args.out else Path(args.run_dirs[0]).parent / "l0_vs_mse.png"
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

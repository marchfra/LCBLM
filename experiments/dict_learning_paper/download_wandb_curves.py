"""Download per-component loss curves from W&B for a sweep output dir and plot them.

The training loops log every loss term per epoch to W&B (e.g. for VAEE:
train/vaee_{total,recon,cond_kl,sparsity,entropy,ortho} and the val/ mirror) but
results.json only keeps recon / L0 / total. This script pulls the full breakdown
back from W&B so we can see which term dominates the objective.

Runs are matched to the sweep by (a) the run-label names of the output subdirs and
(b) the dir's timestamp (W&B stores created_at in UTC; dir names are local time).

Usage:
    python experiments/dict_learning_paper/download_wandb_curves.py \
        experiments/dict_learning_paper/outputs/sweep_synthetic_5atom_k1_diag/20260608-214626
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import wandb

# Dir names are written in local time; W&B created_at is UTC. CEST = UTC+2.
_LOCAL_UTC_OFFSET_H = 2
_PROJECT = "dict-learning-paper"


def _dir_time_utc(run_dir: Path) -> datetime:
    """Parse the YYYYMMDD-HHMMSS dir name (local) and return UTC datetime."""
    stamp = run_dir.name.split("_")[0]  # tolerate suffixes like _bs32
    local = datetime.strptime(stamp, "%Y%m%d-%H%M%S")
    return local.replace(tzinfo=timezone.utc) - timedelta(hours=_LOCAL_UTC_OFFSET_H)


def _expected_names(run_dir: Path) -> dict[str, str]:
    """Map wandb run name -> output subdir label, from each results.json."""
    names: dict[str, str] = {}
    for path in sorted(run_dir.glob("*/results.json")):
        data = json.loads(path.read_text())
        model = data["model_name"]
        label = path.parent.name[len(model) + 1 :]
        names[f"{model}_{label}"] = path.parent.name
    return names


def _match_runs(run_dir: Path, project: str):
    """Return {subdir_name: wandb Run} matched by name + closeness in time."""
    target = _dir_time_utc(run_dir)
    expected = _expected_names(run_dir)
    api = wandb.Api()
    # Pull a generous window around the sweep start.
    lo = (target - timedelta(minutes=15)).isoformat()
    hi = (target + timedelta(hours=6)).isoformat()
    runs = api.runs(
        project,
        filters={"createdAt": {"$gte": lo, "$lte": hi}},
        order="+created_at",
    )
    matched: dict[str, object] = {}
    for r in runs:
        if r.name in expected and expected[r.name] not in matched:
            matched[expected[r.name]] = r
    missing = set(expected.values()) - set(matched)
    if missing:
        print(f"WARNING: no W&B run matched for: {sorted(missing)}")
    return matched


def _history(run) -> dict[str, list[float]]:
    """Full per-epoch history of all train/ and val/ scalar keys."""
    keys: dict[str, list[float]] = {}
    for row in run.scan_history():
        for k, v in row.items():
            if (k.startswith(("train/", "val/"))) and isinstance(v, (int, float)):
                keys.setdefault(k, []).append(float(v))
    return keys


# Longest first so vaee_se is stripped before vaee. These are the W&B log
# prefixes used by the training loops (not the model_name strings).
_MODEL_PREFIXES = (
    "vaee_se",
    "vaee",
    "topk_sae",
    "topk",
    "sae_concept",
    "sae",
    "vq_vae",
    "vqvae",
    "beta_vae",
    "betavae",
)


def _term(key: str) -> str:
    """train/vaee_shared_encoder_cond_kl -> cond_kl (strip split + model prefix)."""
    tail = key.split("/", 1)[1]
    for pref in _MODEL_PREFIXES:
        if tail.startswith(pref + "_"):
            return tail[len(pref) + 1 :]
    return tail


def plot(run_dir: Path, project: str) -> None:
    matched = _match_runs(run_dir, project)
    if not matched:
        print("No runs matched — nothing to plot.")
        return

    histories = {name: _history(r) for name, r in matched.items()}
    # Union of loss terms across all runs, in a stable order.
    preferred = [
        "total",
        "recon",
        "cond_kl",
        "sparsity",
        "entropy",
        "ortho",
        "kl",
        "aux",
    ]
    terms: list[str] = []
    for h in histories.values():
        for k in h:
            t = _term(k)
            if t not in terms:
                terms.append(t)
    terms.sort(key=lambda t: (preferred.index(t) if t in preferred else 99, t))

    n_rows, n_cols = len(matched), len(terms)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.4 * n_rows), squeeze=False
    )
    fig.suptitle(f"W&B loss curves — {run_dir.name}", fontsize=13)

    for row, (name, hist) in enumerate(sorted(histories.items())):
        for col, term in enumerate(terms):
            ax = axes[row][col]
            for split, style in (("train", "-"), ("val", "--")):
                key = next(
                    (k for k in hist if k.startswith(f"{split}/") and _term(k) == term),
                    None,
                )
                if key:
                    ax.plot(hist[key], style, linewidth=1.3, label=split)
            if row == 0:
                ax.set_title(term, fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=7)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    out = run_dir / "learning_curves_wandb.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Timestamped sweep output dir")
    p.add_argument("--project", default=_PROJECT)
    args = p.parse_args()
    plot(Path(args.run_dir), args.project)


if __name__ == "__main__":
    main()

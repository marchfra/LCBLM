"""CLI entry point for the VAEE vs SparseAE experiment.

Subcommands
-----------
run   Load a TOML config, train all models, save results and checkpoints, and
      generate plots.
plot  Load a previously saved results JSON and regenerate plots.

Usage
-----
    vaee-exp run  experiments/vaee_vs_sae/config_sst2.toml
    vaee-exp plot experiment_outputs/.../results.json
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import torch

from lcblm.utils import get_device
from lcblm.utils.plotting import set_plt_style
from lcblm.utils.seed import set_seeds

from .exp_config import DatasetConfig, RunConfig
from .exp_data import DATASET_REGISTRY
from .exp_io import load_results, save_config_json, save_results
from .exp_plotting import plot_l0_recon, plot_learning_curves
from .exp_training import run_experiment

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # ty:ignore[unresolved-import]
    except ModuleNotFoundError as e:
        msg = "tomllib requires Python 3.11+. On older versions run: pip install tomli"
        raise ModuleNotFoundError(msg) from e


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_run_config(raw: dict) -> RunConfig:
    """Build a RunConfig from a config-file dict, ignoring unknown keys."""
    known = set(RunConfig.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in known}
    others = {k: v for k, v in raw.items() if k not in known}
    if not all(k.startswith("_") for k in others):
        unknown = [k for k in others if not k.startswith("_")]
        msg = f"Unknown config keys (prefix with '_' to treat as comments): {unknown}"
        raise ValueError(msg)
    return RunConfig(**filtered, device=get_device())


def _out_dir(base_dir: Path, ds_cfg: DatasetConfig, run_cfg: RunConfig) -> Path:
    p = base_dir / ds_cfg.name / f"epochs_{run_cfg.epochs}" / f"lr_{run_cfg.lr:.0e}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _checkpoint_path(out_dir: Path, model_name: str, n_concepts: int) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir / f"{model_name.lower().replace('-', '_')}_{n_concepts:03d}.pt"


def _scaler_path(out_dir: Path) -> Path:
    return out_dir / "scaler.pkl"


# ── Subcommands ───────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    dataset_name: str = raw.pop("dataset", "")
    if dataset_name not in DATASET_REGISTRY:
        msg = f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY)}"
        raise ValueError(msg)

    ds_cfg, load_data = DATASET_REGISTRY[dataset_name]

    # Apply TOML overrides for DatasetConfig fields
    for key in ("embeddings_path", "input_dim", "eos_token_id", "n_samples"):
        if key in raw:
            ds_cfg = replace(ds_cfg, **{key: raw.pop(key)})

    run_cfg = _load_run_config(raw)
    set_seeds(run_cfg.seed)

    print(f"Dataset : {ds_cfg.name}  ({ds_cfg.input_dim} dims)")
    print(f"Device  : {run_cfg.device}")

    train_ds, val_ds, scaler = load_data(ds_cfg)
    print(
        f"Train sentences: {train_ds.num_sentences}  Val sentences: {val_ds.num_sentences}\n",  # noqa: E501
    )

    results, trained_models = run_experiment(train_ds, val_ds, run_cfg, ds_cfg)

    base = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).parent / "experiment_outputs"
    )
    out_dir = _out_dir(base, ds_cfg, run_cfg)

    save_results(results, run_cfg, ds_cfg, out_dir / "results.json")
    save_config_json(run_cfg, ds_cfg, out_dir / "config.json")

    with _scaler_path(out_dir).open("wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {_scaler_path(out_dir).name}")

    for model_name, n_concepts, model in trained_models:
        ckpt_path = _checkpoint_path(out_dir, model_name, n_concepts)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint {ckpt_path.name}")

    if not args.no_plots:
        print("\nGenerating plots...")
        set_plt_style(["grid", "science", "notebook", "mylegend"], "mplstyles")
        plot_l0_recon(results, run_cfg, ds_cfg, out_dir)
        plot_learning_curves(results, run_cfg, out_dir)

    print(f"\nAll done — outputs in {out_dir}")


def cmd_plot(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    results, run_cfg, ds_cfg = load_results(results_path)
    out_dir = results_path.parent

    print(f"Loaded {len(results)} run(s) from {results_path}")
    print(f"Dataset: {ds_cfg.name}")
    print("Generating plots...")

    set_plt_style(["grid", "science", "notebook", "mylegend"], args.mplstyles)
    plot_l0_recon(results, run_cfg, ds_cfg, out_dir)
    plot_learning_curves(results, run_cfg, out_dir)

    print(f"\nAll done — outputs in {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vaee-exp",
        description="VAEE vs SparseAE comparison experiment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Train models from a TOML config file.")
    run_parser.add_argument("config", help="Path to the run config TOML file.")
    run_parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip all plots.",
    )
    run_parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="",
        help="Base output directory (default: experiment_outputs/ next to this file).",
    )

    plot_parser = sub.add_parser(
        "plot",
        help="Regenerate plots from a saved results JSON.",
    )
    plot_parser.add_argument("results", help="Path to the results JSON file.")
    plot_parser.add_argument(
        "--mplstyles",
        "-mpls",
        type=str,
        default="mplstyles",
        help="Path to the matplotlib stylesheet folder.",
    )

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()

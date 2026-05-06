"""CLI entry point for the VAEE vs SparseAE experiment.

Subcommands
-----------
run     Load a TOML config, train one or all model types, save results and
        checkpoints.
export  Read one or more results JSON files and write a CSV summary suitable
        for plotting (columns: name, l0, mse).
plot    (Coming soon) Generate L0-vs-MSE scatter from a CSV produced by export.

Usage
-----
    vaee-exp run  --model vaee        experiments/vaee_vs_sae/config_sst2.toml
    vaee-exp run  --model sae_concept experiments/vaee_vs_sae/config_sst2.toml
    vaee-exp run  --model sae_param   experiments/vaee_vs_sae/config_sst2.toml
    vaee-exp run  experiments/vaee_vs_sae/config_sst2.toml  # all models

    vaee-exp export dir_vaee/results.json dir_sae_c/results.json dir_sae_p/results.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from experiments.vaee_vs_sae.exp_config import DatasetConfig, RunConfig
from experiments.vaee_vs_sae.exp_data import DATASET_REGISTRY
from experiments.vaee_vs_sae.exp_io import load_results, save_config_json, save_results
from experiments.vaee_vs_sae.exp_training import _run_name, run_experiment
from lcblm.utils import get_device
from lcblm.utils.seed import set_seeds

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # ty:ignore[unresolved-import]
    except ModuleNotFoundError as e:
        msg = "tomllib requires Python 3.11+. On older versions run: pip install tomli"
        raise ModuleNotFoundError(msg) from e


# Fields never treated as grid axes — they have their own semantics or are
# runtime-determined.
_EXCLUDED_FROM_GRID = frozenset(
    {
        "num_embeddings_list",
        "device",
        "skip_vaee",
        "skip_sae_concept_matched",
        "skip_sae_param_matched",
        "wandb_project",
    },
)

# Maps --model values to the skip_* flags that suppress the other model types.
_MODEL_SKIP_FLAGS: dict[str, dict[str, bool]] = {
    "vaee": {"skip_sae_concept_matched": True, "skip_sae_param_matched": True},
    "sae_concept": {"skip_vaee": True, "skip_sae_param_matched": True},
    "sae_param": {"skip_vaee": True, "skip_sae_concept_matched": True},
}


# ── Grid expansion helpers ────────────────────────────────────────────────────


def _combos(axes: dict[str, list]) -> list[dict]:
    """Return the cartesian product of axes as a list of dicts."""
    if not axes:
        return [{}]
    keys = list(axes)
    return [
        dict(zip(keys, vals, strict=False))
        for vals in itertools.product(*axes.values())
    ]


def _expand_grid(  # noqa: C901
    filtered: dict,
    user_skip: dict,
) -> tuple[list[RunConfig], list[str]]:
    """Partition fields into sweep axes and fixed values, then expand.

    Fields starting with 'vaee_' that have list values are VAEE-only axes;
    'sae_' prefixed list fields are SAE-only axes; other list fields are shared
    axes (retraining all models for each value).

    To avoid redundant training, per-model axes are expanded independently:

    - Only vaee axes swept  -> one VAEE-only run per vaee combo + one SAE-only
                               baseline run (per shared combo).
    - Only sae axes swept   -> one VAEE-only baseline + one SAE-only run per
                               sae combo (per shared combo).
    - Both swept            -> vaee-only runs x vaee combos + sae-only runs x
                               sae combos (per shared combo).
    - Neither swept         -> one joint run (all models) per shared combo.

    Explicit skip_* flags in user_skip override the auto-skip logic.
    """
    vaee_axes: dict[str, list] = {}
    sae_axes: dict[str, list] = {}
    shared_axes: dict[str, list] = {}
    fixed: dict = {}

    for k, v in filtered.items():
        if k in _EXCLUDED_FROM_GRID:
            fixed[k] = v
            continue
        if isinstance(v, list):
            if k.startswith("vaee_"):
                vaee_axes[k] = v
            elif k.startswith("sae_"):
                sae_axes[k] = v
            else:
                shared_axes[k] = v
        else:
            fixed[k] = v

    vaee_combos = _combos(vaee_axes)
    sae_combos = _combos(sae_axes)
    shared_combos = _combos(shared_axes)
    grid_keys = list(vaee_axes) + list(sae_axes) + list(shared_axes)

    has_vaee_sweep = bool(vaee_axes)
    has_sae_sweep = bool(sae_axes)

    def _make(
        extra: dict,
        *,
        skip_vaee: bool = False,
        skip_sae: bool = False,
    ) -> RunConfig:
        params = {**fixed, **extra}
        if skip_vaee and "skip_vaee" not in user_skip:
            params["skip_vaee"] = True
        if skip_sae:
            if "skip_sae_concept_matched" not in user_skip:
                params["skip_sae_concept_matched"] = True
            if "skip_sae_param_matched" not in user_skip:
                params["skip_sae_param_matched"] = True
        params.update(user_skip)
        return RunConfig(**params, device=get_device())

    configs: list[RunConfig] = []

    for shared in shared_combos:
        if not has_vaee_sweep and not has_sae_sweep:
            configs.append(_make(shared))
        elif has_vaee_sweep and not has_sae_sweep:
            configs.extend(_make({**shared, **v}, skip_sae=True) for v in vaee_combos)
            configs.append(_make(shared, skip_vaee=True))
        elif not has_vaee_sweep and has_sae_sweep:
            configs.append(_make(shared, skip_sae=True))
            configs.extend(_make({**shared, **s}, skip_vaee=True) for s in sae_combos)
        else:
            configs.extend(_make({**shared, **v}, skip_sae=True) for v in vaee_combos)
            configs.extend(_make({**shared, **s}, skip_vaee=True) for s in sae_combos)

    return configs, grid_keys


def _load_run_configs(raw: dict) -> tuple[list[RunConfig], list[str]]:
    """Build RunConfigs from a config-file dict, expanding list values as grid axes.

    Returns (configs, grid_key_names) where grid_key_names lists all swept fields
    in order (used for progress logging).
    """
    known = set(RunConfig.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in known}
    others = {k: v for k, v in raw.items() if k not in known}
    if not all(k.startswith("_") for k in others):
        unknown = [k for k in others if not k.startswith("_")]
        msg = f"Unknown config keys (prefix with '_' to treat as comments): {unknown}"
        raise ValueError(msg)

    user_skip = {
        k: filtered[k]
        for k in ("skip_vaee", "skip_sae_concept_matched", "skip_sae_param_matched")
        if k in filtered
    }

    return _expand_grid(filtered, user_skip)


# ── Path helpers ──────────────────────────────────────────────────────────────


def _out_dir(base_dir: Path, group: str) -> Path:
    p = base_dir / group
    p.mkdir(parents=True, exist_ok=True)
    return p


def _checkpoint_path(out_dir: Path, run_name: str) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir / f"{run_name}.pt"


def _scaler_path(out_dir: Path) -> Path:
    return out_dir / "scaler.pkl"


# ── Subcommands ───────────────────────────────────────────────────────────────


def _save_run_outputs(  # noqa: PLR0913
    results: list,
    trained_models: list,
    group: str,
    run_cfg: RunConfig,
    ds_cfg: DatasetConfig,
    scaler: StandardScaler,
    base: Path,
) -> Path:
    """Save results, checkpoints, and scaler for one grid point."""
    out_dir = _out_dir(base, group)

    save_results(results, run_cfg, ds_cfg, out_dir / "results.json")
    save_config_json(run_cfg, ds_cfg, out_dir / "config.json")

    with _scaler_path(out_dir).open("wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {_scaler_path(out_dir).name}")

    for run_name, model in trained_models:
        ckpt_path = _checkpoint_path(out_dir, run_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint {ckpt_path.name}")

    return out_dir


def cmd_run(args: argparse.Namespace) -> None:  # noqa: C901
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

    for key in ("embeddings_path", "input_dim", "eos_token_id", "n_samples"):
        if key in raw:
            ds_cfg = replace(ds_cfg, **{key: raw.pop(key)})

    # --model injects skip_* flags before grid expansion.
    if args.model is not None:
        raw.update(_MODEL_SKIP_FLAGS[args.model])

    run_cfgs, grid_keys = _load_run_configs(raw)
    n_cfgs = len(run_cfgs)

    print(f"Dataset : {ds_cfg.name}  ({ds_cfg.input_dim} dims)")
    print(f"Device  : {run_cfgs[0].device}")
    if args.model:
        print(f"Model   : {args.model}")
    if n_cfgs > 1:
        print(f"Grid    : {n_cfgs} configurations over [{', '.join(grid_keys)}]")

    train_ds, val_ds, scaler = load_data(ds_cfg)
    print(
        f"Train sentences: {train_ds.num_sentences}"
        f"  Val sentences: {val_ds.num_sentences}\n",
    )

    base = (
        Path(args.out_dir)
        if args.out_dir
        else Path(__file__).parent / "experiment_outputs"
    )

    last_out_dir: Path | None = None

    for i, run_cfg in enumerate(run_cfgs, 1):
        if n_cfgs > 1:
            grid_vals = "  ".join(f"{k}={getattr(run_cfg, k)}" for k in grid_keys)
            print(f"[{i}/{n_cfgs}]  {grid_vals}")

        set_seeds(run_cfg.seed)

        results, trained_models, group = run_experiment(
            train_ds,
            val_ds,
            run_cfg,
            ds_cfg,
        )

        out_dir = _save_run_outputs(
            results,
            trained_models,
            group,
            run_cfg,
            ds_cfg,
            scaler,
            base,
        )
        last_out_dir = out_dir

        if n_cfgs > 1:
            print(f"  -> {out_dir}\n")

    if n_cfgs > 1:
        print(f"All done — {n_cfgs} configurations completed.")
    else:
        print(f"\nAll done — outputs in {last_out_dir}")


def cmd_export(args: argparse.Namespace) -> None:
    """Write a CSV summary of best_l0 and best_val_recon for each run."""
    paths = [Path(p) for p in args.results]
    for p in paths:
        if not p.exists():
            print(f"Error: results file not found: {p}", file=sys.stderr)
            sys.exit(1)

    rows: list[dict[str, object]] = []
    for p in paths:
        results, run_cfg, _ = load_results(p)
        for r in results:
            name = _run_name(r.model_name, r.sweep_n, r.n_concepts, run_cfg)
            rows.append({"name": name, "l0": r.best_l0, "mse": r.best_val_recon})

    out_path = Path(args.out) if args.out else paths[0].parent / "summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "l0", "mse"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")


def cmd_plot(args: argparse.Namespace) -> None:  # noqa: ARG001
    print(
        "vaee-exp plot is not yet implemented.\n"
        "Use `vaee-exp export` to produce a CSV, then plot from that.",
        file=sys.stderr,
    )
    sys.exit(1)


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
        "--model",
        choices=["vaee", "sae_concept", "sae_param"],
        default=None,
        help="Train only this model type (default: train all three).",
    )
    run_parser.add_argument(
        "--out-dir",
        "-o",
        type=str,
        default="",
        help="Base output directory (default: experiment_outputs/ next to this file).",
    )

    export_parser = sub.add_parser(
        "export",
        help="Export best_l0 and best_val_recon from results JSON files to a CSV.",
    )
    export_parser.add_argument(
        "results",
        nargs="+",
        help="One or more paths to results.json files.",
    )
    export_parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path (default: summary.csv next to the first results file).",
    )

    plot_parser = sub.add_parser(
        "plot",
        help="(Coming soon) Generate L0-vs-MSE scatter from a CSV.",
    )
    plot_parser.add_argument("csv", help="Path to a CSV produced by vaee-exp export.")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "plot":
        cmd_plot(args)


if __name__ == "__main__":
    main()

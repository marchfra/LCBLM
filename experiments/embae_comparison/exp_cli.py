# ruff: noqa: N806

"""CLI entry point for the SparseAE vs EmbeddingAE experiment.

Subcommands
-----------
run   Load a TOML config, train all models, save results and checkpoints, and
      generate plots.
plot  Load a previously saved results JSON, optionally load checkpoints to
      regenerate concept dictionary plots.

Usage
-----
    python exp_cli.py run  config_digits.toml
    python exp_cli.py run  config_mnist_14x14.toml
    python exp_cli.py plot experiment_outputs/digits/lambda_1e-02/results.json
    python exp_cli.py plot experiment_outputs/digits/lambda_1e-02/results.json --no-concept-dicts

Requires Python 3.11+ for tomllib (stdlib). On older versions install tomli:
    pip install tomli
"""  # noqa: E501

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import time
from datetime import timedelta

from .exp_training import build_embedding_ae, build_sparse_ae, run_experiment

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # ty:ignore[unresolved-import]
    except ModuleNotFoundError as e:
        msg = "tomllib requires Python 3.11+. On older versions run: pip install tomli"
        raise ModuleNotFoundError(msg) from e
from pathlib import Path

import torch
from torch import nn

from lcblm.utils.plotting import set_plt_style
from lcblm.utils.seed import set_seeds

from .exp_config import DatasetConfig, RunConfig
from .exp_data import DATASET_REGISTRY
from .exp_io import load_results, save_config_json, save_results
from .exp_plotting import (
    plot_concept_ablation,
    plot_concept_dictionary,
    plot_concept_reconstructions,
    plot_l0_recon,
    plot_learning_curves,
    plot_per_sample_mse_boxplot,
)
from .exp_sweep import RunIndex, parse_sweep_config

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_run_config(raw: dict) -> RunConfig:
    """Build a RunConfig from a config-file dict, ignoring unknown keys."""
    known = set(RunConfig.__dataclass_fields__)
    filtered = {k: v for k, v in raw.items() if k in known}
    return RunConfig(
        **filtered,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def _out_dir(ds_cfg: DatasetConfig, run_cfg: RunConfig) -> Path:
    if run_cfg.sparsity_mode == "l1":
        sparsity_dir = Path("l1") / f"lambda_l1_{run_cfg.lambda_l1:.0e}"
    else:
        sparsity_dir = (
            Path("kl") / f"p_{run_cfg.target_p}" / f"lambda_kl_{run_cfg.lambda_kl:.0e}"
        )
    p = (
        Path(__file__).parent
        / "experiment_outputs"
        / ds_cfg.name
        / sparsity_dir
        / f"tau_{run_cfg.tau:.0e}"
        / f"mu_{run_cfg.mu:.0e}"
    )
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _checkpoint_path(out_dir: Path, model_name: str, n_concepts: int) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir / f"{model_name.lower()}_{n_concepts:03d}.pt"


def _scaler_path(out_dir: Path) -> Path:
    return out_dir / "scaler.pkl"


def _build_model(
    model_name: str,
    n_concepts: int,
    run_cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> nn.Module:
    """Reconstruct an untrained model with the correct architecture."""
    if model_name == "SparseAE":
        return build_sparse_ae(n_concepts, run_cfg, ds_cfg)
    if model_name == "EmbeddingAE":
        return build_embedding_ae(n_concepts, run_cfg, ds_cfg)

    msg = f"Unknown model name: {model_name!r}"
    raise ValueError(msg)


# ── Subcommands ───────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    dataset_name: str = raw.get("dataset", "")
    if dataset_name not in DATASET_REGISTRY:
        print(
            f"Error: unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY)}",
            file=sys.stderr,
        )
        sys.exit(1)

    ds_cfg, load_data = DATASET_REGISTRY[dataset_name]
    run_cfg = _load_run_config(raw)

    set_plt_style(["grid", "science", "notebook", "mylegend"], "mplstyles")
    set_seeds(run_cfg.seed)

    print(f"Dataset : {ds_cfg.name}  ({ds_cfg.input_dim} dims)")
    print(f"Device  : {run_cfg.device}")

    X_train, X_test, _y_train, _y_test, scaler = load_data(ds_cfg.n_samples)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")

    results, trained_models = run_experiment(X_train, X_test, run_cfg, ds_cfg)

    out_dir = _out_dir(ds_cfg, run_cfg)

    # Persist results and checkpoints
    save_results(results, run_cfg, ds_cfg, out_dir / "results.json")
    with _scaler_path(out_dir).open("wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {_scaler_path(out_dir).name}")
    for model_name, n_concepts, model in trained_models:
        ckpt_path = _checkpoint_path(out_dir, model_name, n_concepts)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint {ckpt_path.name}")

    print("\nGenerating plots...")
    plot_l0_recon(results, run_cfg, ds_cfg, out_dir)
    plot_learning_curves(results, run_cfg, out_dir)
    plot_per_sample_mse_boxplot(trained_models, X_train, run_cfg, ds_cfg, out_dir)
    for model_name, n_concepts, model in trained_models:
        plot_concept_dictionary(
            model,
            model_name,
            n_concepts,
            X_train,
            scaler,
            run_cfg,
            ds_cfg,
            out_dir,
        )
        plot_concept_ablation(
            model,
            model_name,
            n_concepts,
            X_train,
            scaler,
            run_cfg,
            ds_cfg,
            out_dir,
        )
        plot_concept_reconstructions(
            model,
            model_name,
            n_concepts,
            X_train,
            scaler,
            run_cfg,
            ds_cfg,
            out_dir,
        )

    print(f"\nAll done — outputs in {out_dir}")


def cmd_plot(args: argparse.Namespace) -> None:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    results, run_cfg, ds_cfg = load_results(results_path)
    out_dir = results_path.parent  # plots go next to the results file

    set_plt_style(["grid", "science", "notebook", "mylegend"], "../../mplstyles/")

    print(f"Loaded {len(results)} run(s) from {results_path}")
    print(f"Dataset: {ds_cfg.name}")
    print("Generating plots...")

    plot_l0_recon(results, run_cfg, ds_cfg, out_dir)
    plot_learning_curves(results, run_cfg, out_dir)

    if args.no_concept_dicts:
        print("Skipping concept dictionary plots (--no-concept-dicts).")
        print(f"\nAll done — outputs in {out_dir}")
        return

    # Concept dictionaries need activations → reload data and model checkpoints
    if ds_cfg.name not in DATASET_REGISTRY:
        print(
            f"Warning: dataset '{ds_cfg.name}' not in registry; "
            "cannot generate concept dictionaries.",
            file=sys.stderr,
        )
        return

    print("\nLoading data and checkpoints for concept dictionaries...")
    _, load_data = DATASET_REGISTRY[ds_cfg.name]
    X_train, _X_test, _y_train, _y_test, fresh_scaler = load_data(ds_cfg.n_samples)
    scaler_path = _scaler_path(out_dir)
    if scaler_path.exists():
        with scaler_path.open("rb") as f:
            scaler = pickle.load(f)  # noqa: S301
    else:
        print(
            "Warning: scaler not found, refitting from data (results may differ slightly).",  # noqa: E501
        )
        scaler = fresh_scaler

    # Re-run boxplot using reloaded checkpoints
    plot_mse_models: list[tuple[str, int, nn.Module]] = []
    for result in results:
        model_name = result.model_name
        n_concepts = result.n_concepts
        ckpt_path = _checkpoint_path(out_dir, model_name, n_concepts)

        if not ckpt_path.exists():
            print(
                f"Warning: checkpoint not found for {model_name} n={n_concepts} "
                f"({ckpt_path.name}), skipping.",
                file=sys.stderr,
            )
            continue

        model = _build_model(model_name, n_concepts, run_cfg, ds_cfg)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=run_cfg.device, weights_only=True),
        )
        plot_mse_models.append((model_name, n_concepts, model))
        plot_concept_dictionary(
            model,
            model_name,
            n_concepts,
            X_train,
            scaler,
            run_cfg,
            ds_cfg,
            out_dir,
        )
        plot_concept_ablation(
            model,
            model_name,
            n_concepts,
            X_train,
            scaler,
            run_cfg,
            ds_cfg,
            out_dir,
        )
        plot_concept_reconstructions(
            model,
            model_name,
            n_concepts,
            X_train,
            scaler,
            run_cfg,
            ds_cfg,
            out_dir,
        )
    plot_per_sample_mse_boxplot(plot_mse_models, X_train, run_cfg, ds_cfg, out_dir)

    print(f"\nAll done — outputs in {out_dir}")


def cmd_sweep(args: argparse.Namespace) -> None:
    sweep_path = Path(args.config)
    if not sweep_path.exists():
        print(f"Error: sweep config not found: {sweep_path}", file=sys.stderr)
        sys.exit(1)

    ds_cfg, run_cfgs = parse_sweep_config(sweep_path)

    base_dir = Path(__file__).parent / "experiment_outputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    index = RunIndex(base_dir / "sweep_index.json")

    print(f"Sweep: {len(run_cfgs)} run(s) over cartesian product")
    print(f"Dataset: {ds_cfg.name}  ({ds_cfg.input_dim} dims)")
    print(f"Device:  {run_cfgs[0].device}\n")

    set_plt_style(["grid", "science", "notebook", "mylegend"], "../../mplstyles/")

    _, load_data = DATASET_REGISTRY[ds_cfg.name]
    X_train, X_test, _y_train, _y_test, scaler = load_data(ds_cfg.n_samples)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")

    for i, run_cfg in enumerate(run_cfgs, 1):
        start_time = time.time()

        # Check for existing identical run
        existing = index.find_existing(run_cfg, ds_cfg)
        if existing is not None:
            print(f"[{i}/{len(run_cfgs)}] Skipping — identical to {existing}")
            continue

        run_id = index.next_run_id()
        out_dir = base_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(run_cfgs)}] Starting {run_id}")
        set_seeds(run_cfg.seed)

        # Save config before training so partial runs are inspectable
        save_config_json(run_cfg, ds_cfg, out_dir / "config.json")

        with _scaler_path(out_dir).open("wb") as f:
            pickle.dump(scaler, f)
        print(f"Saved scaler to {_scaler_path(out_dir).name}")

        results, trained_models = run_experiment(X_train, X_test, run_cfg, ds_cfg)

        save_results(results, run_cfg, ds_cfg, out_dir / "results.json")
        for model_name, n_concepts, model in trained_models:
            ckpt_path = _checkpoint_path(out_dir, model_name, n_concepts)
            torch.save(model.state_dict(), ckpt_path)

        print(f"  Generating plots for {run_id}...")
        plot_l0_recon(results, run_cfg, ds_cfg, out_dir)
        plot_learning_curves(results, run_cfg, out_dir)
        plot_per_sample_mse_boxplot(trained_models, X_train, run_cfg, ds_cfg, out_dir)
        for model_name, n_concepts, model in trained_models:
            plot_concept_dictionary(
                model,
                model_name,
                n_concepts,
                X_train,
                scaler,
                run_cfg,
                ds_cfg,
                out_dir,
            )
            plot_concept_ablation(
                model,
                model_name,
                n_concepts,
                X_train,
                scaler,
                run_cfg,
                ds_cfg,
                out_dir,
            )
            plot_concept_reconstructions(
                model,
                model_name,
                n_concepts,
                X_train,
                scaler,
                run_cfg,
                ds_cfg,
                out_dir,
            )

        # Register and save index after each completed run so a crash midway
        # doesn't lose track of finished runs
        index.register(run_id, run_cfg, ds_cfg)
        index.save()
        print(
            f"  Done in {timedelta(seconds=round(time.time() - start_time))} — "
            f"outputs in ./{out_dir.relative_to(Path.cwd())}/\n",
        )

    print(
        f"Sweep complete. Index at ./{(base_dir / 'sweep_index.json').relative_to(Path.cwd())}",  # noqa: E501
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="exp_cli",
        description="SparseAE vs EmbeddingAE comparison experiment.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Train models from a TOML config file.")
    run_p.add_argument("config", help="Path to the run config TOML file.")

    plot_p = sub.add_parser("plot", help="Regenerate plots from a saved results JSON.")
    plot_p.add_argument("results", help="Path to the results JSON file.")
    plot_p.add_argument(
        "--no-concept-dicts",
        action="store_true",
        default=False,
        help="Skip concept dictionary plots (avoids loading checkpoints and data).",
    )

    sweep_p = sub.add_parser(
        "sweep",
        help="Run a cartesian product sweep from a sweep config.",
    )
    sweep_p.add_argument("config", help="Path to the sweep config TOML file.")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "sweep":
        cmd_sweep(args)


if __name__ == "__main__":
    main()

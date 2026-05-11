"""Training CLI for the concept_training experiment.

Trains one concept model at a time on pre-extracted LLM token embeddings:
  vaee        — variational autoencoder with discrete prototype gates
  topk_sae    — sparse autoencoder with TopK activation (no L1 penalty)
  sae_concept — L1 SAE whose latent_dim matches a VAEE's num_embeddings
  sae_param   — L1 SAE whose parameter count matches a VAEE

Supports single-run and multi-run (sequential) TOML configs. Multi-run mode
uses TOML array-of-tables (``[[runs]]``): shared fields live at the top level
and each entry overrides only the keys it specifies.

Output directories are named after key hyperparameters plus a timestamp:
  VAEE-{num_embeddings}x{embedding_size}-{timestamp}
  TopK-{k}-SAE-{latent_dim}-{timestamp}
  L1-{lambda_l1}-SAE-{latent_dim}-{timestamp}

Usage
-----
    ct-train vaee        experiments/concept_training/configs/vaee_sst2.toml
    ct-train topk_sae    experiments/concept_training/configs/topk_sae_sst2.toml
    ct-train sae_concept experiments/concept_training/configs/sae_concept_sst2.toml
    ct-train sae_param   experiments/concept_training/configs/sae_param_sst2.toml
    ct-train vaee        experiments/concept_training/configs/vaee_multi.toml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from sklearn.preprocessing import StandardScaler  # noqa: TC002
from torch import nn

import wandb
from lcblm.training.configs import (
    MODEL_CONFIG_CLASSES,
    DatasetConfig,
    SAEConceptConfig,
    SAEParamConfig,
    TopKSAEConfig,
    VAEEConfig,
    _BaseConfig,
)
from lcblm.training.data import load_embeddings, save_scaler
from lcblm.training.loops import (
    train_sae_concept,
    train_sae_param,
    train_topk_sae,
    train_vaee,
)
from lcblm.training.models import build_ref_vaee, param_matched_latent_dim
from lcblm.utils import get_device, set_seeds
from lcblm.utils.data import NextTokenDataset  # noqa: TC001

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # ty:ignore[unresolved-import]


AnyConfig = VAEEConfig | TopKSAEConfig | SAEConceptConfig | SAEParamConfig

_CFG_TO_MODEL_TYPE: dict[type, str] = {
    VAEEConfig: "vaee",
    TopKSAEConfig: "topk_sae",
    SAEConceptConfig: "sae_concept",
    SAEParamConfig: "sae_param",
}


# ── Config loading ────────────────────────────────────────────────────────────


def _make_config(cfg_cls: type, raw: dict) -> AnyConfig:
    known = set(cfg_cls.__dataclass_fields__) | set(_BaseConfig.__dataclass_fields__)  # ty:ignore[unresolved-attribute]
    run_raw = {k: v for k, v in raw.items() if k in known}
    unknown = [
        k for k in raw if k not in known and not k.startswith("_") and k != "run_name"
    ]
    if unknown:
        msg = (
            f"Unknown config keys for {cfg_cls.__name__} "
            f"(prefix with '_' to suppress): {unknown}"
        )
        raise ValueError(msg)
    return cfg_cls(**run_raw, device=get_device())


def _load_configs(
    toml_path: Path,
    model_type: str,
) -> tuple[list[tuple[AnyConfig, str]], DatasetConfig]:
    """Parse a TOML config; return ([(cfg, run_name), ...], dataset_config).

    Supports both single-run (no [[runs]] key) and multi-run ([[runs]] array).
    The run_name is an optional user-supplied output-dir override; empty string
    means auto-generate from hyperparameters.
    """
    with toml_path.open("rb") as f:
        raw = tomllib.load(f)

    ds_fields = set(DatasetConfig.__dataclass_fields__)
    ds_raw = {k: raw.pop(k) for k in list(raw) if k in ds_fields}
    if not ds_raw.get("name"):
        ds_raw["name"] = toml_path.stem
    ds_cfg = DatasetConfig(**ds_raw)

    cfg_cls = MODEL_CONFIG_CLASSES[model_type]
    runs_list: list[dict] | None = raw.pop("runs", None)

    if runs_list is None:
        run_name = raw.pop("run_name", "")
        cfg = _make_config(cfg_cls, raw)
        return [(cfg, run_name)], ds_cfg

    shared = {k: v for k, v in raw.items() if not k.startswith("_")}
    result = []
    for run_raw in runs_list:
        run_name = run_raw.pop("run_name", "")
        cfg = _make_config(cfg_cls, {**shared, **run_raw})
        result.append((cfg, run_name))
    return result, ds_cfg


# ── Output dir naming ─────────────────────────────────────────────────────────


def _auto_run_name(cfg: AnyConfig, latent_dim: int) -> str:
    if isinstance(cfg, VAEEConfig):
        return f"VAEE-{cfg.num_embeddings}x{cfg.embedding_size}"
    if isinstance(cfg, TopKSAEConfig):
        return f"TopK-{cfg.k}-SAE-{latent_dim}"
    if isinstance(cfg, (SAEConceptConfig, SAEParamConfig)):
        return f"L1-{cfg.lambda_l1}-SAE-{latent_dim}"
    msg = f"Unknown config type: {type(cfg)}"
    raise TypeError(msg)


def _resolve_latent_dim(cfg: AnyConfig, input_dim: int) -> int:
    if isinstance(cfg, VAEEConfig):
        return 0
    if isinstance(cfg, TopKSAEConfig):
        return cfg.latent_dim if cfg.latent_dim > 0 else 4 * input_dim
    if isinstance(cfg, SAEConceptConfig):
        return cfg.vaee_num_embeddings
    if isinstance(cfg, SAEParamConfig):
        ref_vaee = build_ref_vaee(input_dim, cfg)
        return param_matched_latent_dim(ref_vaee, input_dim)
    msg = f"Unknown config type: {type(cfg)}"
    raise TypeError(msg)


# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def _save_checkpoint(
    model: nn.Module,
    name: str,
    metadata: dict,
    ckpt_dir: Path,
) -> None:
    torch.save(model.state_dict(), ckpt_dir / f"{name}.pt")
    with (ckpt_dir / f"{name}_meta.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    print(
        f"  Saved {name}.pt  (L0={metadata.get('best_l0', '?'):.2f}"
        f"  val_MSE={metadata.get('best_val_recon', '?'):.5f})",
    )


# ── Single run ────────────────────────────────────────────────────────────────


def _run_one(  # noqa: PLR0913
    cfg: AnyConfig,
    run_name_override: str,
    ds_cfg: DatasetConfig,
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    scaler: StandardScaler,
    base: Path,
) -> Path:
    model_type = _CFG_TO_MODEL_TYPE[type(cfg)]
    input_dim = train_ds.embedding_dimension
    latent_dim = _resolve_latent_dim(cfg, input_dim)
    auto_name = _auto_run_name(cfg, latent_dim)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{run_name_override or auto_name}-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_scaler(scaler, out_dir / "scaler.pkl")

    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict.pop("device")
    with (out_dir / "config.json").open("w") as f:
        json.dump(
            {
                "model_type": model_type,
                "dataset_config": dataclasses.asdict(ds_cfg),
                "run_config": cfg_dict,
            },
            f,
            indent=2,
        )

    wandb_run = None
    if cfg.wandb_project:
        cfg_log = {**cfg_dict, **dataclasses.asdict(ds_cfg), "model_type": model_type}
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=auto_name,
            config=cfg_log,
        )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if isinstance(cfg, VAEEConfig):
        model, result = train_vaee(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": model_type,
            "input_dim": input_dim,
            "num_embeddings": cfg.num_embeddings,
            "embedding_size": cfg.embedding_size,
            "encoder_type": cfg.encoder_type,
            "topology": cfg.topology,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }
    elif isinstance(cfg, TopKSAEConfig):
        model, result = train_topk_sae(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": model_type,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "topk_k": cfg.k,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }
    elif isinstance(cfg, SAEConceptConfig):
        model, result = train_sae_concept(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": model_type,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }
    elif isinstance(cfg, SAEParamConfig):
        model, result = train_sae_param(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": model_type,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }
    else:
        msg = f"Unknown config type: {type(cfg)}"
        raise TypeError(msg)

    if wandb_run is not None:
        wandb_run.finish()

    result.run_name = auto_name
    _save_checkpoint(model, auto_name, metadata, ckpt_dir)

    with (out_dir / "results.json").open("w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2)

    print(f"  -> {out_dir}")
    return out_dir


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ct-train",
        description="Train concept models on pre-extracted LLM embeddings.",
    )
    parser.add_argument(
        "model",
        choices=list(MODEL_CONFIG_CLASSES),
        help="Model type to train.",
    )
    parser.add_argument("config", help="Path to the TOML config file.")
    parser.add_argument(
        "--out-dir",
        "-o",
        default="",
        help="Base output directory (default: outputs/ next to this file).",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    run_configs, ds_cfg = _load_configs(config_path, args.model)
    set_seeds(run_configs[0][0].seed)

    print(f"Model   : {args.model}")
    print(f"Dataset : {ds_cfg.name}")
    print(f"Device  : {run_configs[0][0].device}")
    if len(run_configs) > 1:
        print(f"Runs    : {len(run_configs)}")

    train_ds, val_ds, scaler = load_embeddings(
        ds_cfg.embeddings_path,
        ds_cfg.eos_token_id,
        ds_cfg.n_samples,
    )
    print(
        f"Train: {train_ds.num_sentences} sentences  "
        f"Val: {val_ds.num_sentences} sentences  "
        f"Embedding dim: {train_ds.embedding_dimension}\n",
    )

    base = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "outputs"

    for i, (cfg, run_name_override) in enumerate(run_configs, 1):
        if len(run_configs) > 1:
            print(f"[{i}/{len(run_configs)}]")
        set_seeds(cfg.seed)
        _run_one(cfg, run_name_override, ds_cfg, train_ds, val_ds, scaler, base)

    if len(run_configs) > 1:
        print(f"\nAll done — {len(run_configs)} runs completed.")
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()

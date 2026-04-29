"""Training CLI for the interpretability experiment.

Trains four model variants on pre-extracted LLM token embeddings:
  1. VAEE          — variational autoencoder with discrete prototype gates
  2. TopK SAE      — sparse autoencoder with TopK activation (no L1 penalty)
  3. SAE-concept   — L1 SAE with latent_dim matching VAEE's num_embeddings
  4. SAE-param     — L1 SAE with parameter count matching VAEE

All checkpoints, metadata, and the fitted scaler are saved to a timestamped
output directory for use by build_cd.py and intervene.py.

Usage
-----
    interp-train run experiments/interpretability/configs/sst2.toml
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import trange

import wandb
from experiments.interpretability.data import load_embeddings, save_scaler
from lcblm.sae_utils import SparseAE, TopK
from lcblm.sae_utils.dataset import compute_tied_bias
from lcblm.utils import get_device, set_seeds
from lcblm.utils.data import NextTokenDataset, typed_dataloader
from lcblm.vaee.models import VAEE, compute_loss

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # ty:ignore[unresolved-import]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunConfig:
    epochs: int
    lr: float
    batch_size: int = 512
    seed: int = 42
    device: torch.device = field(default_factory=get_device)

    vaee_num_embeddings: int = 256
    vaee_embedding_size: int = 128
    vaee_hidden_dim: int = 256
    vaee_encoder_type: Literal["mlp", "linear", "shallow"] = "shallow"
    vaee_gumbel_temp: float = 0.5
    vaee_sigma_0: float = 0.0
    vaee_sim_metric: Literal["cosine", "inner_product", "neg_euclidean"] = "cosine"
    vaee_topology: Literal["stacked", "summed"] = "stacked"
    vaee_pi: float = 0.1
    vaee_gamma: float = 0.01
    vaee_beta: float = 1.0
    vaee_beta_warmup_epochs: int = 0
    vaee_lambda_ent: float = 0.01
    vaee_lambda_ortho: float = 0.0

    # TopK SAE: k active features per token; latent_dim = 0 → 4 * input_dim
    topk_k: int = 64
    topk_latent_dim: int = 0

    sae_lambda_l1: float = 1e-3
    sae_normalize_decoder: bool = True

    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    skip_vaee: bool = False
    skip_topk_sae: bool = False
    skip_sae_concept: bool = False
    skip_sae_param: bool = False

    wandb_project: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    embeddings_path: str
    input_dim: int
    eos_token_id: int
    n_samples: int = -1


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    model_name: str
    n_concepts: int
    train_recon: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    train_l0: list[float] = field(default_factory=list)
    val_l0: list[float] = field(default_factory=list)
    best_val_recon: float = float("inf")
    best_l0: float = float("inf")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _flat_tokens(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return embeddings[mask]


def _early_stop(result: RunResult, cfg: RunConfig) -> bool:
    if (
        cfg.early_stopping_patience <= 0
        or len(result.val_recon) < cfg.early_stopping_patience
    ):
        return False
    recent = result.val_recon[-cfg.early_stopping_patience :]
    return recent[0] - min(recent) <= cfg.early_stopping_min_delta


# ── Model builders ────────────────────────────────────────────────────────────


def _build_vaee(cfg: RunConfig, ds_cfg: DatasetConfig) -> VAEE:
    return VAEE(
        input_dim=ds_cfg.input_dim,
        hidden_dim=cfg.vaee_hidden_dim,
        num_embeddings=cfg.vaee_num_embeddings,
        embedding_size=cfg.vaee_embedding_size,
        gumbel_temp=cfg.vaee_gumbel_temp,
        output_activation=None,
        encoder_type=cfg.vaee_encoder_type,
        sigma_0=cfg.vaee_sigma_0,
        sim_metric=cfg.vaee_sim_metric,
        topology=cfg.vaee_topology,
    ).to(cfg.device)


def _build_sae(
    latent_dim: int,
    activation: nn.Module,
    train_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
) -> SparseAE:
    model = SparseAE(
        input_dim=ds_cfg.input_dim,
        latent_dim=latent_dim,
        activation=activation,
    ).to(cfg.device)
    flat = train_ds.embeddings[train_ds.attention_mask].cpu()
    model.init_tied_bias(compute_tied_bias(flat, sample_every=15))
    return model


def _param_matched_latent_dim(vaee: VAEE, input_dim: int) -> int:
    vaee_params = sum(p.numel() for p in vaee.parameters() if p.requires_grad)
    return max(1, round((vaee_params - input_dim) / (2 * input_dim)))


# ── Training loops ────────────────────────────────────────────────────────────


def _train_vaee(
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[VAEE, RunResult]:
    model = _build_vaee(cfg, ds_cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="VAEE", n_concepts=cfg.vaee_num_embeddings)

    for epoch in trange(cfg.epochs, desc="VAEE", unit="epoch"):
        warmup = cfg.vaee_beta_warmup_epochs
        effective_beta = cfg.vaee_beta * (
            min(1.0, epoch / warmup) if warmup > 0 else 1.0
        )

        model.train()
        t_recon = t_l0 = t_count = 0.0
        for batch in typed_dataloader(train_loader):
            tokens = _flat_tokens(
                batch.embeddings.to(cfg.device),
                batch.attention_mask.to(cfg.device),
            )
            out = model(tokens)
            decoder_weight = (
                model.decoder_first_weight()
                if cfg.vaee_topology == "stacked" and cfg.vaee_lambda_ortho > 0
                else None
            )
            loss_out = compute_loss(
                target=tokens,
                input=out.recon,
                mu=out.mu,
                alpha=out.alpha,
                prototypes=model.prototypes,
                pi=cfg.vaee_pi,
                gamma=cfg.vaee_gamma,
                beta=effective_beta,
                lambda_ent=cfg.vaee_lambda_ent,
                lambda_ortho=cfg.vaee_lambda_ortho,
                decoder_weight=decoder_weight,
                num_embeddings=model.num_embeddings,
                embedding_size=model.embedding_size,
            )
            optimizer.zero_grad()
            loss_out.total_loss.backward()
            optimizer.step()
            t_recon += loss_out.recon_loss.item()
            t_l0 += (out.c > 1e-6).float().sum(dim=1).sum().item()  # noqa: PLR2004
            t_count += out.c.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(t_recon / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        v_recon = v_l0 = v_count = 0.0
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = _flat_tokens(
                    batch.embeddings.to(cfg.device),
                    batch.attention_mask.to(cfg.device),
                )
                out = model(tokens)
                loss_out = compute_loss(
                    target=tokens,
                    input=out.recon,
                    mu=out.mu,
                    alpha=out.alpha,
                    prototypes=model.prototypes,
                    pi=cfg.vaee_pi,
                    gamma=cfg.vaee_gamma,
                    beta=effective_beta,
                    lambda_ent=cfg.vaee_lambda_ent,
                    lambda_ortho=cfg.vaee_lambda_ortho,
                    num_embeddings=model.num_embeddings,
                    embedding_size=model.embedding_size,
                )
                v_recon += loss_out.recon_loss.item()
                v_l0 += (out.c > 1e-6).float().sum(dim=1).sum().item()  # noqa: PLR2004
                v_count += out.c.shape[0]

        n_va = len(val_loader)
        val_recon = v_recon / n_va
        result.val_recon.append(val_recon)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/recon": result.train_recon[-1],
                    "train/l0": result.train_l0[-1],
                    "val/recon": val_recon,
                    "val/l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        if val_recon < result.best_val_recon - cfg.early_stopping_min_delta:
            result.best_val_recon = val_recon
            result.best_l0 = result.val_l0[-1]
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
        elif _early_stop(result, cfg):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, result


def _train_sae(  # noqa: PLR0913
    model_name: str,
    latent_dim: int,
    activation: nn.Module,
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    *,
    use_l1: bool,
) -> tuple[SparseAE, RunResult]:
    model = _build_sae(latent_dim, activation, train_ds, cfg, ds_cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name=model_name, n_concepts=latent_dim)

    for epoch in trange(cfg.epochs, desc=model_name, unit="epoch"):
        model.train()
        t_recon = t_l0 = t_count = 0.0
        for batch in typed_dataloader(train_loader):
            tokens = _flat_tokens(
                batch.embeddings.to(cfg.device),
                batch.attention_mask.to(cfg.device),
            )
            out = model(tokens)
            recon_loss = F.mse_loss(out.recon, tokens)
            loss = (
                recon_loss + cfg.sae_lambda_l1 * out.latents.abs().mean()
                if use_l1
                else recon_loss
            )
            optimizer.zero_grad()
            loss.backward()
            if cfg.sae_normalize_decoder:
                model.project_decoder_gradients()
            optimizer.step()
            if cfg.sae_normalize_decoder:
                model.normalize_decoder()
            t_recon += recon_loss.item()
            t_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
            t_count += out.latents.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(t_recon / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        v_recon = v_l0 = v_count = 0.0
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = _flat_tokens(
                    batch.embeddings.to(cfg.device),
                    batch.attention_mask.to(cfg.device),
                )
                out = model(tokens)
                v_recon += F.mse_loss(out.recon, tokens).item()
                v_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
                v_count += out.latents.shape[0]

        n_va = len(val_loader)
        val_recon = v_recon / n_va
        result.val_recon.append(val_recon)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/recon": result.train_recon[-1],
                    "train/l0": result.train_l0[-1],
                    "val/recon": val_recon,
                    "val/l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        if val_recon < result.best_val_recon - cfg.early_stopping_min_delta:
            result.best_val_recon = val_recon
            result.best_l0 = result.val_l0[-1]
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
        elif _early_stop(result, cfg):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    return model, result


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


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_experiment(  # noqa: C901
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: RunConfig,
    ds_cfg: DatasetConfig,
    out_dir: Path,
) -> list[RunResult]:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    group = f"{ds_cfg.name}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    if cfg.wandb_project:
        print(f"W&B project: {cfg.wandb_project}  group: {group}")

    def _wandb(name: str, extra: dict) -> wandb.sdk.wandb_run.Run | None:
        if not cfg.wandb_project:
            return None
        cfg_dict = dataclasses.asdict(cfg)
        cfg_dict.pop("device")
        return wandb.init(
            project=cfg.wandb_project,
            group=group,
            name=name,
            config={**cfg_dict, **dataclasses.asdict(ds_cfg), **extra},
            reinit=True,
        )

    results: list[RunResult] = []
    vaee: VAEE | None = None

    if not cfg.skip_vaee:
        print(
            f"\n-- VAEE | num_embeddings={cfg.vaee_num_embeddings} "
            f"embedding_size={cfg.vaee_embedding_size} --",
        )
        run = _wandb("vaee", {"model_type": "vaee"})
        vaee, result = _train_vaee(train_ds, val_ds, cfg, ds_cfg, run)
        if run:
            run.finish()
        results.append(result)
        _save_checkpoint(
            vaee,
            "vaee",
            {
                "model_type": "vaee",
                "input_dim": ds_cfg.input_dim,
                "num_embeddings": cfg.vaee_num_embeddings,
                "embedding_size": cfg.vaee_embedding_size,
                "hidden_dim": cfg.vaee_hidden_dim,
                "encoder_type": cfg.vaee_encoder_type,
                "best_val_recon": result.best_val_recon,
                "best_l0": result.best_l0,
            },
            ckpt_dir,
        )

    if not cfg.skip_topk_sae:
        topk_latent = (
            cfg.topk_latent_dim if cfg.topk_latent_dim > 0 else 4 * ds_cfg.input_dim
        )
        print(f"\n-- TopK SAE | k={cfg.topk_k} latent_dim={topk_latent} --")
        run = _wandb("topk_sae", {"model_type": "topk_sae", "topk_k": cfg.topk_k})
        model, result = _train_sae(
            "TopK-SAE",
            topk_latent,
            TopK(cfg.topk_k),
            use_l1=False,
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg,
            ds_cfg=ds_cfg,
            wandb_run=run,
        )
        if run:
            run.finish()
        results.append(result)
        _save_checkpoint(
            model,
            "topk_sae",
            {
                "model_type": "topk_sae",
                "input_dim": ds_cfg.input_dim,
                "latent_dim": topk_latent,
                "topk_k": cfg.topk_k,
                "best_val_recon": result.best_val_recon,
                "best_l0": result.best_l0,
            },
            ckpt_dir,
        )

    if not cfg.skip_sae_concept:
        latent_dim = cfg.vaee_num_embeddings
        print(f"\n-- SAE-concept | latent_dim={latent_dim} --")
        run = _wandb("sae_concept", {"model_type": "sae_concept"})
        model, result = _train_sae(
            "SAE-concept",
            latent_dim,
            nn.ReLU(),
            use_l1=True,
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg,
            ds_cfg=ds_cfg,
            wandb_run=run,
        )
        if run:
            run.finish()
        results.append(result)
        _save_checkpoint(
            model,
            "sae_concept",
            {
                "model_type": "sae_concept",
                "input_dim": ds_cfg.input_dim,
                "latent_dim": latent_dim,
                "best_val_recon": result.best_val_recon,
                "best_l0": result.best_l0,
            },
            ckpt_dir,
        )

    if not cfg.skip_sae_param:
        ref_vaee = vaee if vaee is not None else _build_vaee(cfg, ds_cfg)
        latent_dim = _param_matched_latent_dim(ref_vaee, ds_cfg.input_dim)
        if vaee is None:
            del ref_vaee
        print(f"\n-- SAE-param | latent_dim={latent_dim} (param-matched to VAEE) --")
        run = _wandb("sae_param", {"model_type": "sae_param"})
        model, result = _train_sae(
            "SAE-param",
            latent_dim,
            nn.ReLU(),
            use_l1=True,
            train_ds=train_ds,
            val_ds=val_ds,
            cfg=cfg,
            ds_cfg=ds_cfg,
            wandb_run=run,
        )
        if run:
            run.finish()
        results.append(result)
        _save_checkpoint(
            model,
            "sae_param",
            {
                "model_type": "sae_param",
                "input_dim": ds_cfg.input_dim,
                "latent_dim": latent_dim,
                "best_val_recon": result.best_val_recon,
                "best_l0": result.best_l0,
            },
            ckpt_dir,
        )

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def _load_config(toml_path: Path) -> tuple[RunConfig, DatasetConfig]:
    with toml_path.open("rb") as f:
        raw = tomllib.load(f)

    ds_fields = {"name", "embeddings_path", "input_dim", "eos_token_id", "n_samples"}
    ds_raw = {k: raw.pop(k) for k in list(raw) if k in ds_fields}
    if "name" not in ds_raw:
        ds_raw["name"] = toml_path.stem

    known_run = set(RunConfig.__dataclass_fields__)
    run_raw = {k: v for k, v in raw.items() if k in known_run}
    unknown = [k for k in raw if k not in known_run and not k.startswith("_")]
    if unknown:
        msg = f"Unknown config keys (prefix with '_' to comment out): {unknown}"
        raise ValueError(msg)

    return RunConfig(**run_raw, device=get_device()), DatasetConfig(**ds_raw)


def cmd_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg, ds_cfg = _load_config(config_path)
    set_seeds(cfg.seed)

    print(f"Dataset : {ds_cfg.name}  ({ds_cfg.input_dim} dims)")
    print(f"Device  : {cfg.device}")

    train_ds, val_ds, scaler = load_embeddings(
        ds_cfg.embeddings_path,
        ds_cfg.input_dim,
        ds_cfg.eos_token_id,
        ds_cfg.n_samples,
    )
    print(
        f"Train: {train_ds.num_sentences} sentences  "
        f"Val: {val_ds.num_sentences} sentences\n",
    )

    base = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "outputs"
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{ds_cfg.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_scaler(scaler, out_dir / "scaler.pkl")
    with (out_dir / "config.json").open("w") as f:
        cfg_dict = dataclasses.asdict(cfg)
        cfg_dict.pop("device")
        json.dump(
            {"dataset_config": dataclasses.asdict(ds_cfg), "run_config": cfg_dict},
            f,
            indent=2,
        )

    results = run_experiment(train_ds, val_ds, cfg, ds_cfg, out_dir)

    results_payload = [dataclasses.asdict(r) for r in results]
    with (out_dir / "results.json").open("w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nAll done — outputs in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="interp-train",
        description="Train concept models for interpretability analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Train from a TOML config.")
    run_p.add_argument("config", help="Path to the TOML config file.")
    run_p.add_argument(
        "--out-dir",
        "-o",
        default="",
        help="Base output directory (default: outputs/ next to this file).",
    )

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()

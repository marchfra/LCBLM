"""Training CLI for the interpretability experiment.

Trains one concept model at a time on pre-extracted LLM token embeddings:
  vaee        — variational autoencoder with discrete prototype gates
  topk_sae    — sparse autoencoder with TopK activation (no L1 penalty)
  sae_concept — L1 SAE whose latent_dim matches the VAEE's num_embeddings
  sae_param   — L1 SAE whose parameter count matches the VAEE

All checkpoints, metadata, and the fitted scaler are saved to a timestamped
output directory for use by build_cd.py and intervene.py.

Usage
-----
    interp-train run --model vaee        experiments/interpretability/configs/vaee_sst2.toml
    interp-train run --model topk_sae    experiments/interpretability/configs/topk_sae_sst2.toml
    interp-train run --model sae_concept experiments/interpretability/configs/sae_concept_sst2.toml
    interp-train run --model sae_param   experiments/interpretability/configs/sae_param_sst2.toml
"""  # noqa: E501

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
class _BaseConfig:
    epochs: int
    lr: float
    batch_size: int = 512
    seed: int = 42
    device: torch.device = field(default_factory=get_device)
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    wandb_project: str | None = None


@dataclass(frozen=True)
class VAEEConfig(_BaseConfig):
    num_embeddings: int = 256
    embedding_size: int = 128
    hidden_dim: int = 256  # only used when encoder_type = "mlp"
    encoder_type: Literal["mlp", "linear", "shallow"] = "shallow"
    gumbel_temp: float = 0.5
    sigma_0: float = 0.1
    sim_metric: Literal["cosine", "inner_product", "neg_euclidean"] = "cosine"
    topology: Literal["stacked", "summed"] = "stacked"
    pi: float = 0.1
    gamma: float = 0.01
    beta: float = 1.0
    lambda_ent: float = 0.01
    lambda_ortho: float = 1e-3


@dataclass(frozen=True)
class TopKSAEConfig(_BaseConfig):
    k: int = 64
    latent_dim: int = 0  # 0 → 4 * input_dim
    normalize_decoder: bool = True


@dataclass(frozen=True)
class SAEConceptConfig(_BaseConfig):
    vaee_num_embeddings: int = 256  # latent_dim = this value
    lambda_l1: float = 1e-3
    normalize_decoder: bool = True


@dataclass(frozen=True)
class SAEParamConfig(_BaseConfig):
    # VAEE reference architecture — used only to count parameters
    vaee_num_embeddings: int = 256
    vaee_embedding_size: int = 128
    vaee_hidden_dim: int = 256  # only used when vaee_encoder_type = "mlp"
    vaee_encoder_type: Literal["mlp", "linear", "shallow"] = "shallow"
    vaee_topology: Literal["stacked", "summed"] = "stacked"
    lambda_l1: float = 1e-3
    normalize_decoder: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    embeddings_path: str
    eos_token_id: int
    n_samples: int = -1
    name: str = ""  # defaults to config filename stem


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


def _early_stop(result: RunResult, patience: int, min_delta: float) -> bool:
    if patience <= 0 or len(result.val_recon) < patience:
        return False
    recent = result.val_recon[-patience:]
    return recent[0] - min(recent) <= min_delta


# ── Model builders ────────────────────────────────────────────────────────────


def _build_vaee(input_dim: int, cfg: VAEEConfig) -> VAEE:
    return VAEE(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        num_embeddings=cfg.num_embeddings,
        embedding_size=cfg.embedding_size,
        gumbel_temp=cfg.gumbel_temp,
        output_activation=None,
        encoder_type=cfg.encoder_type,
        sigma_0=cfg.sigma_0,
        sim_metric=cfg.sim_metric,
        topology=cfg.topology,
    ).to(cfg.device)


def _build_ref_vaee(input_dim: int, cfg: SAEParamConfig) -> VAEE:
    """Instantiate a VAEE reference model for parameter counting only (not trained)."""
    return VAEE(
        input_dim=input_dim,
        hidden_dim=cfg.vaee_hidden_dim,
        num_embeddings=cfg.vaee_num_embeddings,
        embedding_size=cfg.vaee_embedding_size,
        gumbel_temp=0.5,
        output_activation=None,
        encoder_type=cfg.vaee_encoder_type,
        sigma_0=0.1,
        sim_metric="cosine",
        topology=cfg.vaee_topology,
    )


def _build_sae(
    input_dim: int,
    latent_dim: int,
    activation: nn.Module,
    train_ds: NextTokenDataset,
    device: torch.device,
) -> SparseAE:
    model = SparseAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        activation=activation,
    ).to(device)
    flat = train_ds.embeddings[train_ds.attention_mask].cpu()
    model.init_tied_bias(compute_tied_bias(flat, sample_every=15))
    return model


def _param_matched_latent_dim(ref_vaee: VAEE, input_dim: int) -> int:
    vaee_params = sum(p.numel() for p in ref_vaee.parameters() if p.requires_grad)
    return max(1, round((vaee_params - input_dim) / (2 * input_dim)))


# ── Training loops ────────────────────────────────────────────────────────────


def train_vaee(  # noqa: PLR0915
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: VAEEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[VAEE, RunResult]:
    input_dim = train_ds.embedding_dimension
    model = _build_vaee(input_dim, cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    result = RunResult(model_name="vaee", n_concepts=cfg.num_embeddings)
    best_state: dict = {}

    for epoch in trange(cfg.epochs, desc="VAEE", unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "cond_kl": 0.0,
            "sparsity": 0.0,
            "entropy": 0.0,
            "ortho": 0.0,
        }
        t_l0 = t_count = 0.0
        for batch in typed_dataloader(train_loader):
            tokens = _flat_tokens(
                batch.embeddings.to(cfg.device),
                batch.attention_mask.to(cfg.device),
            )
            out = model(tokens)
            decoder_weight = (
                model.decoder_first_weight()
                if cfg.topology == "stacked" and cfg.lambda_ortho > 0
                else None
            )
            loss_out = compute_loss(
                target=tokens,
                input=out.recon,
                mu=out.mu,
                alpha=out.alpha,
                prototypes=model.prototypes,
                pi=cfg.pi,
                gamma=cfg.gamma,
                beta=cfg.beta,
                lambda_ent=cfg.lambda_ent,
                lambda_ortho=cfg.lambda_ortho,
                decoder_weight=decoder_weight,
                num_embeddings=model.num_embeddings,
                embedding_size=model.embedding_size,
            )
            optimizer.zero_grad()
            loss_out.total_loss.backward()
            optimizer.step()
            epoch_terms["total"] += loss_out.total_loss.item()
            epoch_terms["recon"] += loss_out.recon_loss.item()
            epoch_terms["cond_kl"] += loss_out.cond_kl_loss.item()
            epoch_terms["sparsity"] += loss_out.sparsity_loss.item()
            epoch_terms["entropy"] += loss_out.entropy_loss.item()
            epoch_terms["ortho"] += loss_out.ortho_loss.item()
            with torch.no_grad():
                t_l0 += (out.c > 1e-6).float().sum(dim=1).sum().item()  # noqa: PLR2004
                t_count += out.c.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        val_terms: dict[str, float] = {
            "total": 0.0,
            "recon": 0.0,
            "cond_kl": 0.0,
            "sparsity": 0.0,
            "entropy": 0.0,
            "ortho": 0.0,
        }
        v_l0 = v_count = 0.0
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
                    pi=cfg.pi,
                    gamma=cfg.gamma,
                    beta=cfg.beta,
                    lambda_ent=cfg.lambda_ent,
                    lambda_ortho=cfg.lambda_ortho,
                    decoder_weight=(
                        model.decoder_first_weight()
                        if cfg.topology == "stacked" and cfg.lambda_ortho > 0
                        else None
                    ),
                    num_embeddings=model.num_embeddings,
                    embedding_size=model.embedding_size,
                )
                val_terms["total"] += loss_out.total_loss.item()
                val_terms["recon"] += loss_out.recon_loss.item()
                val_terms["cond_kl"] += loss_out.cond_kl_loss.item()
                val_terms["sparsity"] += loss_out.sparsity_loss.item()
                val_terms["entropy"] += loss_out.entropy_loss.item()
                val_terms["ortho"] += loss_out.ortho_loss.item()
                v_l0 += (out.c > 1e-6).float().sum(dim=1).sum().item()  # noqa: PLR2004
                v_count += out.c.shape[0]

        n_va = len(val_loader)
        val_recon = val_terms["recon"] / n_va
        result.val_recon.append(val_recon)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            wandb_run.log(
                {f"train/vaee_{k}": v / n_tr for k, v in epoch_terms.items()}
                | {f"val/vaee_{k}": v / n_va for k, v in val_terms.items()}
                | {
                    "train/vaee_l0": result.train_l0[-1],
                    "val/vaee_l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        if val_recon < result.best_val_recon - cfg.early_stopping_min_delta:
            result.best_val_recon = val_recon
            result.best_l0 = result.val_l0[-1]
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            if wandb_run is not None:
                wandb_run.summary.update(
                    {
                        "best_val_recon": result.best_val_recon,
                        "best_l0": result.best_l0,
                    },
                )
        elif _early_stop(
            result,
            cfg.early_stopping_patience,
            cfg.early_stopping_min_delta,
        ):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, result


def _train_sae(  # noqa: PLR0913, PLR0915
    model_name: str,
    latent_dim: int,
    activation: nn.Module,
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    lambda_l1: float = 0.0,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    *,
    use_l1: bool,
    normalize_decoder: bool,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.embedding_dimension
    model = _build_sae(input_dim, latent_dim, activation, train_ds, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    result = RunResult(model_name=model_name, n_concepts=latent_dim)
    best_state: dict = {}

    for epoch in trange(epochs, desc=model_name, unit="epoch"):
        model.train()
        epoch_terms: dict[str, float] = {"recon": 0.0, "l1": 0.0}
        t_l0 = t_count = 0.0
        for batch in typed_dataloader(train_loader):
            tokens = _flat_tokens(
                batch.embeddings.to(device),
                batch.attention_mask.to(device),
            )
            out = model(tokens)
            recon_loss = F.mse_loss(out.recon, tokens)
            l1_loss = out.latents.abs().mean()
            loss = recon_loss + lambda_l1 * l1_loss if use_l1 else recon_loss
            optimizer.zero_grad()
            loss.backward()
            if normalize_decoder:
                model.project_decoder_gradients()
            optimizer.step()
            if normalize_decoder:
                model.normalize_decoder()
            epoch_terms["recon"] += recon_loss.item()
            epoch_terms["l1"] += l1_loss.item()
            with torch.no_grad():
                t_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
                t_count += out.latents.shape[0]

        n_tr = len(train_loader)
        result.train_recon.append(epoch_terms["recon"] / n_tr)
        result.train_l0.append(t_l0 / t_count)

        model.eval()
        val_terms: dict[str, float] = {"recon": 0.0, "l1": 0.0}
        v_l0 = v_count = 0.0
        with torch.inference_mode():
            for batch in typed_dataloader(val_loader):
                tokens = _flat_tokens(
                    batch.embeddings.to(device),
                    batch.attention_mask.to(device),
                )
                out = model(tokens)
                val_terms["recon"] += F.mse_loss(out.recon, tokens).item()
                val_terms["l1"] += out.latents.abs().mean().item()
                v_l0 += (out.latents > 0).float().sum(dim=1).sum().item()
                v_count += out.latents.shape[0]

        n_va = len(val_loader)
        val_recon = val_terms["recon"] / n_va
        result.val_recon.append(val_recon)
        result.val_l0.append(v_l0 / v_count)

        if wandb_run is not None:
            wandb_run.log(
                {f"train/sae_{k}": v / n_tr for k, v in epoch_terms.items()}
                | {f"val/sae_{k}": v / n_va for k, v in val_terms.items()}
                | {
                    "train/sae_l0": result.train_l0[-1],
                    "val/sae_l0": result.val_l0[-1],
                },
                step=epoch + 1,
            )

        if val_recon < result.best_val_recon - early_stopping_min_delta:
            result.best_val_recon = val_recon
            result.best_l0 = result.val_l0[-1]
            best_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
            if wandb_run is not None:
                wandb_run.summary.update(
                    {
                        "best_val_recon": result.best_val_recon,
                        "best_l0": result.best_l0,
                    },
                )
        elif _early_stop(result, early_stopping_patience, early_stopping_min_delta):
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)  # type: ignore[arg-type]
    model.eval()
    return model, result


def train_topk_sae(
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: TopKSAEConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.embedding_dimension
    latent_dim = cfg.latent_dim if cfg.latent_dim > 0 else 4 * input_dim
    return _train_sae(
        model_name="topk_sae",
        latent_dim=latent_dim,
        activation=TopK(cfg.k),
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        device=cfg.device,
        normalize_decoder=cfg.normalize_decoder,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_min_delta=cfg.early_stopping_min_delta,
        use_l1=False,
        wandb_run=wandb_run,
    )


def train_sae_concept(
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: SAEConceptConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    return _train_sae(
        model_name="sae_concept",
        latent_dim=cfg.vaee_num_embeddings,
        activation=nn.ReLU(),
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        device=cfg.device,
        normalize_decoder=cfg.normalize_decoder,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_min_delta=cfg.early_stopping_min_delta,
        lambda_l1=cfg.lambda_l1,
        use_l1=True,
        wandb_run=wandb_run,
    )


def train_sae_param(
    train_ds: NextTokenDataset,
    val_ds: NextTokenDataset,
    cfg: SAEParamConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[SparseAE, RunResult]:
    input_dim = train_ds.embedding_dimension
    ref_vaee = _build_ref_vaee(input_dim, cfg)
    latent_dim = _param_matched_latent_dim(ref_vaee, input_dim)
    del ref_vaee
    return _train_sae(
        model_name="sae_param",
        latent_dim=latent_dim,
        activation=nn.ReLU(),
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        device=cfg.device,
        normalize_decoder=cfg.normalize_decoder,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_min_delta=cfg.early_stopping_min_delta,
        lambda_l1=cfg.lambda_l1,
        use_l1=True,
        wandb_run=wandb_run,
    )


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


# ── Config loading ────────────────────────────────────────────────────────────

_MODEL_CONFIG_CLASSES = {
    "vaee": VAEEConfig,
    "topk_sae": TopKSAEConfig,
    "sae_concept": SAEConceptConfig,
    "sae_param": SAEParamConfig,
}


def _load_config(
    toml_path: Path,
    model_type: str,
) -> tuple[
    VAEEConfig | TopKSAEConfig | SAEConceptConfig | SAEParamConfig,
    DatasetConfig,
]:
    with toml_path.open("rb") as f:
        raw = tomllib.load(f)

    ds_fields = set(DatasetConfig.__dataclass_fields__)
    ds_raw = {k: raw.pop(k) for k in list(raw) if k in ds_fields}
    if not ds_raw.get("name"):
        ds_raw["name"] = toml_path.stem

    cfg_cls = _MODEL_CONFIG_CLASSES[model_type]
    known = set(cfg_cls.__dataclass_fields__)
    # also accept base fields
    known |= set(_BaseConfig.__dataclass_fields__)
    run_raw = {k: v for k, v in raw.items() if k in known}
    unknown = [k for k in raw if k not in known and not k.startswith("_")]
    if unknown:
        msg = (
            f"Unknown config keys for model '{model_type}' "
            f"(prefix with '_' to suppress): {unknown}"
        )
        raise ValueError(msg)

    return cfg_cls(**run_raw, device=get_device()), DatasetConfig(**ds_raw)


# ── CLI ───────────────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:  # noqa: PLR0915
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    model_type: str = args.model
    cfg, ds_cfg = _load_config(config_path, model_type)
    set_seeds(cfg.seed)

    print(f"Model   : {model_type}")
    print(f"Dataset : {ds_cfg.name}")
    print(f"Device  : {cfg.device}")

    train_ds, val_ds, scaler = load_embeddings(
        ds_cfg.embeddings_path,
        ds_cfg.eos_token_id,
        ds_cfg.n_samples,
    )
    input_dim = train_ds.embedding_dimension
    print(
        f"Train: {train_ds.num_sentences} sentences  "
        f"Val: {val_ds.num_sentences} sentences  "
        f"Embedding dim: {input_dim}\n",
    )

    base = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "outputs"
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{ds_cfg.name}_{model_type}_{timestamp}"
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
            name=f"{ds_cfg.name}_{model_type}",
            config=cfg_log,
        )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "vaee":
        assert isinstance(cfg, VAEEConfig)  # noqa: S101
        model, result = train_vaee(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": "vaee",
            "input_dim": input_dim,
            "num_embeddings": cfg.num_embeddings,
            "embedding_size": cfg.embedding_size,
            "encoder_type": cfg.encoder_type,
            "topology": cfg.topology,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }

    elif model_type == "topk_sae":
        assert isinstance(cfg, TopKSAEConfig)  # noqa: S101
        latent_dim = cfg.latent_dim if cfg.latent_dim > 0 else 4 * input_dim
        model, result = train_topk_sae(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": "topk_sae",
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "topk_k": cfg.k,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }

    elif model_type == "sae_concept":
        assert isinstance(cfg, SAEConceptConfig)  # noqa: S101
        model, result = train_sae_concept(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": "sae_concept",
            "input_dim": input_dim,
            "latent_dim": cfg.vaee_num_embeddings,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }

    elif model_type == "sae_param":
        assert isinstance(cfg, SAEParamConfig)  # noqa: S101
        ref_vaee = _build_ref_vaee(input_dim, cfg)
        latent_dim = _param_matched_latent_dim(ref_vaee, input_dim)
        del ref_vaee
        model, result = train_sae_param(train_ds, val_ds, cfg, wandb_run)
        metadata = {
            "model_type": "sae_param",
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "best_val_recon": result.best_val_recon,
            "best_l0": result.best_l0,
        }
    else:
        msg = f"Unknown model type: {model_type}"
        raise ValueError(msg)

    if wandb_run is not None:
        wandb_run.finish()

    _save_checkpoint(model, model_type, metadata, ckpt_dir)

    with (out_dir / "results.json").open("w") as f:
        json.dump(dataclasses.asdict(result), f, indent=2)

    print(f"\nAll done — outputs in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="interp-train",
        description="Train concept models for interpretability analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Train from a TOML config.")
    run_p.add_argument(
        "--model",
        "-m",
        required=True,
        choices=list(_MODEL_CONFIG_CLASSES),
        help="Model type to train.",
    )
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

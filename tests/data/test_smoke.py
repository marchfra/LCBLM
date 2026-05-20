"""Smoke test: every data-loader type × every model type, 2 training steps.

Image loaders are exercised with mocked torchvision datasets (no downloads).
dSprites is exercised with a synthetic npz written to a tmp_path fixture.
All model configs use tiny hyperparameters so the test runs on CPU in seconds.
"""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lcblm.data.image_loaders import load_dsprites, load_fmnist, load_mnist
from lcblm.data.synthetic import make_synthetic
from lcblm.training.configs import (
    BetaVAEConfig,
    SAEConceptConfig,
    SAEParamConfig,
    TopKSAEConfig,
    VAEEConfig,
    VQVAEConfig,
)
from lcblm.training.loops import (
    train_beta_vae,
    train_sae_concept,
    train_sae_param,
    train_topk_sae,
    train_vaee,
    train_vq_vae,
)
from lcblm.utils.data import FlatTensorDataset

# ── Constants ─────────────────────────────────────────────────────────────────

CPU = torch.device("cpu")
N_TRAIN = 100
N_VAL = 20
BATCH = 50  # N_TRAIN / BATCH = 2 batches → 2 training steps per epoch

# ── Model runner ──────────────────────────────────────────────────────────────


def _run_model(
    name: str, train_ds: FlatTensorDataset, val_ds: FlatTensorDataset
) -> None:
    """Run 1 epoch (2 steps) of the named model on the given datasets."""
    common = {"epochs": 1, "lr": 1e-3, "batch_size": BATCH, "device": CPU}

    if name == "vaee":
        cfg = VAEEConfig(**common, num_embeddings=4, embedding_size=8, pi=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_vaee(train_ds, val_ds, cfg)

    elif name == "topk_sae":
        cfg = TopKSAEConfig(**common, k=2, latent_dim=8, k_aux=4)
        train_topk_sae(train_ds, val_ds, cfg)

    elif name == "sae_concept":
        cfg = SAEConceptConfig(**common, vaee_num_embeddings=4, lambda_l1=0.01)
        train_sae_concept(train_ds, val_ds, cfg)

    elif name == "sae_param":
        cfg = SAEParamConfig(
            **common,
            vaee_num_embeddings=4,
            vaee_embedding_size=8,
            lambda_l1=0.01,
        )
        train_sae_param(train_ds, val_ds, cfg)

    elif name == "vq_vae":
        cfg = VQVAEConfig(**common, num_codes=4)
        train_vq_vae(train_ds, val_ds, cfg)

    elif name == "beta_vae":
        cfg = BetaVAEConfig(**common, latent_dim=4, beta=1.0)
        train_beta_vae(train_ds, val_ds, cfg)

    else:
        msg = f"Unknown model: {name}"
        raise ValueError(msg)


# ── Dataset factories ─────────────────────────────────────────────────────────


def _synth(input_dim: int) -> tuple[FlatTensorDataset, FlatTensorDataset]:
    """Synthetic dataset with the given input_dim."""
    train, _ = make_synthetic(
        n_samples=N_TRAIN, n_features=16, n_active=2, input_dim=input_dim, seed=0
    )
    val, _ = make_synthetic(
        n_samples=N_VAL, n_features=16, n_active=2, input_dim=input_dim, seed=1
    )
    return train, val


def _mock_tv_dataset(n: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.data = torch.zeros(n, 28, 28, dtype=torch.uint8)
    return mock


# ── Parametrised smoke test ───────────────────────────────────────────────────

MODELS = ["vaee", "topk_sae", "sae_concept", "sae_param", "vq_vae", "beta_vae"]


@pytest.mark.parametrize("model_name", MODELS)
def test_synthetic_all_models(model_name):
    train_ds, val_ds = _synth(input_dim=32)
    _run_model(model_name, train_ds, val_ds)


@pytest.mark.parametrize("model_name", MODELS)
@patch("lcblm.data.image_loaders.MNIST")
def test_mnist_all_models(mock_cls, model_name):
    mock_cls.return_value = _mock_tv_dataset()
    train_ds = load_mnist("/tmp/fake", "train", n_samples=N_TRAIN)
    val_ds = load_mnist("/tmp/fake", "val", n_samples=N_VAL)
    _run_model(model_name, train_ds, val_ds)


@pytest.mark.parametrize("model_name", MODELS)
@patch("lcblm.data.image_loaders.FashionMNIST")
def test_fmnist_all_models(mock_cls, model_name):
    mock_cls.return_value = _mock_tv_dataset()
    train_ds = load_fmnist("/tmp/fake", "train", n_samples=N_TRAIN)
    val_ds = load_fmnist("/tmp/fake", "val", n_samples=N_VAL)
    _run_model(model_name, train_ds, val_ds)


@pytest.mark.parametrize("model_name", MODELS)
def test_dsprites_all_models(tmp_path, model_name):
    npz_path = str(tmp_path / "dsprites.npz")
    n_total = N_TRAIN + N_VAL + 10  # enough for 80/20 split
    imgs = np.zeros((n_total, 64, 64), dtype=np.uint8)
    latents = np.zeros((n_total, 6), dtype=np.int64)
    np.savez(npz_path, imgs=imgs, latents_classes=latents)

    train_ds, _ = load_dsprites(npz_path, "train", n_samples=N_TRAIN)
    val_ds, _ = load_dsprites(npz_path, "val", n_samples=N_VAL)
    _run_model(model_name, train_ds, val_ds)

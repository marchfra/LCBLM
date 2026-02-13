import json
from pathlib import Path

import pytest

from src.sae_utils.config import Config


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    config_data = {
        "latent_dim_factor": 8,
        "n_epochs": 250,
        "learning_rate": 1e-5,
        "activation": "topk",
        "k": 128,
        "threshold_dead_latent": 25_000,
        "alpha_aux_loss": 1.0 / 32.0,
        "k_aux": 256,
        "batch_size": 256,
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config_data, f)
    return config_path


def test_config_from_json(config_file: Path) -> None:
    config = Config.from_json(config_file)
    assert config.latent_dim_factor == 8
    assert config.n_epochs == 250
    assert config.learning_rate == 1e-5
    assert config.activation == "topk"
    assert config.k == 128
    assert config.threshold_dead_latent == 25_000
    assert config.alpha_aux_loss == 1.0 / 32.0
    assert config.k_aux == 256
    assert config.batch_size == 256


def test_config_save_to_json(config_file: Path) -> None:
    config = Config.from_json(config_file)
    config.save_to_json(config_file.parent / "saved_config.json")
    with (config_file.parent / "saved_config.json").open("r") as f:
        saved_config = json.load(f)
    assert saved_config == {
        "latent_dim_factor": 8,
        "n_epochs": 250,
        "learning_rate": 1e-5,
        "activation": "topk",
        "k": 128,
        "threshold_dead_latent": 25_000,
        "device": "cuda",
        "alpha_aux_loss": 1.0 / 32.0,
        "k_aux": 256,
        "batch_size": 256,
    }


def test_config_file_not_found(tmp_path: Path) -> None:
    non_existent_file = tmp_path / "non_existent_config.json"
    with pytest.raises(FileNotFoundError):
        Config.from_json(non_existent_file)


def test_config_invalid_json(tmp_path: Path) -> None:
    invalid_json_file = tmp_path / "invalid_config.json"
    with invalid_json_file.open("w") as f:
        f.write("{invalid_json: true,}")  # Invalid JSON format
    with pytest.raises(ValueError, match="Error decoding JSON from"):
        Config.from_json(invalid_json_file)

import sys
import tomllib
from enum import Enum, auto
from pathlib import Path

from pydantic import BaseModel

from lcblm._logging import utils_logger as logger

CONFIG_FILE = Path("config.toml")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding extraction."""

    batch_size: int
    max_length: int = 256


class FinetuneConfig(BaseModel):
    """Configuration for finetuning of the original LLM head."""

    batch_size: int


class LinearTrainingConfig(BaseModel):
    """Configuration for training a linear classifier."""

    batch_size: int
    p_dropout: float = 0
    use_bias: bool = False


class BaselineConfig(BaseModel):
    """Configuration for the baselines."""

    finetune: FinetuneConfig
    train: LinearTrainingConfig


class SAEConfig(BaseModel):
    """Configuration for a Sparse AutoEncoder."""

    batch_size: int
    latent_dim_factor: int = 4


class SAEClfConfig(BaseModel):
    """Configuration for a SAE + linear classifier."""

    sae: SAEConfig
    head: LinearTrainingConfig


class LCBLMConfig(BaseModel):
    """Configuration for a Concept Embedding Model."""


class PerplexityConfig(BaseModel):
    """Configuration for perplexity evaluation."""

    llm: str


class MetricsConfig(BaseModel):
    """Configuration for metrics evaluation."""

    perplexity: PerplexityConfig


class Config(BaseModel):
    """Configuration for the whole project."""

    embedding: EmbeddingConfig
    baseline: BaselineConfig
    sae: SAEClfConfig
    lcblm: LCBLMConfig
    metrics: MetricsConfig
    backbone_llm: str = "mistralai/Mistral-7B-v0.1"
    dataset: str = "sst2"


def read_config(file_path: str | Path = CONFIG_FILE) -> Config:
    """Read a configuration file."""
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("Config file not found", extra={"file_path": file_path})
        sys.exit(1)

    with file_path.open("rb") as f:
        config_dict = tomllib.load(f)

    return Config.model_validate(config_dict)


if __name__ == "__main__":
    config = read_config()
    print(config)

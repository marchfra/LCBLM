import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """Configuration for SAE utils."""

    n_epochs: int
    batch_size: int
    learning_rate: float
    latent_dim_factor: int
    k: int
    threshold_dead_latent: int
    device: Literal["cpu", "cuda"] = "cuda"
    alpha_aux_loss: float = 1 / 32  # In accordance with OpenAI's paper
    activation: Literal["topk"] = "topk"
    k_aux: int = 512  # In accordance with OpenAI's paper

    @classmethod
    def from_json(cls, json_file: Path) -> "Config":
        """Create Config instance from a JSON file."""
        if not json_file.exists():
            raise FileNotFoundError(f"Config file {json_file} does not exist.")

        with json_file.open("r") as f:
            try:
                config_dict = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {json_file}") from e
        return cls(**config_dict)

    def save_to_json(self, json_file: Path) -> None:
        """Save Config instance to a JSON file."""
        with json_file.open("w") as f:
            json.dump(asdict(self), f, indent=2)

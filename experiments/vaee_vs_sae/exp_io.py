"""JSON serialisation and deserialisation of experiment results and configs.

Saved file structure:
    {
        "dataset_config": { ...DatasetConfig fields... },
        "run_config":     { ...RunConfig fields, device excluded... },
        "results":        [ ...RunResult dicts... ]
    }

The device field of RunConfig is always re-detected at load time and never
written to disk, since it is machine-specific.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

from lcblm.utils import get_device

from .exp_config import DatasetConfig, RunConfig
from .exp_training import RunResult

if TYPE_CHECKING:
    from pathlib import Path


def _run_config_to_dict(cfg: RunConfig) -> dict:
    d = dataclasses.asdict(cfg)
    d.pop("device")
    return d


def _run_config_from_dict(d: dict) -> RunConfig:
    return RunConfig(**d, device=get_device())


def save_results(
    results: list[RunResult],
    run_cfg: RunConfig,
    ds_cfg: DatasetConfig,
    path: Path,
) -> None:
    """Serialise results and both configs to a JSON file.

    Args:
        results: RunResult objects returned by run_experiment().
        run_cfg: Hyperparameter configuration used for the run.
        ds_cfg: Dataset metadata used for the run.
        path: Destination file path (parent directory must exist).

    """
    payload = {
        "dataset_config": dataclasses.asdict(ds_cfg),
        "run_config": _run_config_to_dict(run_cfg),
        "results": [dataclasses.asdict(r) for r in results],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {path}")


def load_results(
    path: Path,
) -> tuple[list[RunResult], RunConfig, DatasetConfig]:
    """Load results and configs from a JSON file produced by save_results().

    Args:
        path: Path to the JSON file.

    Returns:
        results, run_cfg, ds_cfg

    """
    with path.open() as f:
        payload = json.load(f)
    ds_cfg = DatasetConfig(**payload["dataset_config"])
    run_cfg = _run_config_from_dict(payload["run_config"])
    results = [RunResult(**r) for r in payload["results"]]
    return results, run_cfg, ds_cfg


def save_config_json(
    run_cfg: RunConfig,
    ds_cfg: DatasetConfig,
    path: Path,
) -> None:
    """Save RunConfig and DatasetConfig as a human-readable JSON file.

    Args:
        run_cfg: Hyperparameter configuration for this run.
        ds_cfg: Dataset metadata for this run.
        path: Destination file path.

    """
    payload = {
        "dataset_config": dataclasses.asdict(ds_cfg),
        "run_config": _run_config_to_dict(run_cfg),
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)

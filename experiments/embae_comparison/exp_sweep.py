"""Sweep config parsing and run index management.

A sweep config is a TOML file with the same fields as a single-run config,
except that any RunConfig field (other than dataset and n_concepts_list) may
be a list of values to sweep over. The cartesian product of all list-valued
fields is computed, producing one RunConfig per combination.

Example sweep config:
    dataset = "digits"
    n_concepts_list = [5, 10, 20, 50]

    epochs = 2000
    lr = [1e-4, 1e-3]
    sparsity_mode = ["l1", "kl"]
    lambda_l1 = [1e-3, 1e-2]
    lambda_kl = 1e-2
    target_p = 0.05
    embedding_size = 8
    batch_size = 128
    seed = 0

This produces 2 x 2 x 2 = 8 runs.
"""

from __future__ import annotations

import copy

from lcblm.utils import get_device

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import itertools
import json
from dataclasses import asdict as _asdict
from typing import TYPE_CHECKING

from .exp_config import RunConfig
from .exp_data import DATASET_REGISTRY

if TYPE_CHECKING:
    from pathlib import Path

    from .exp_config import DatasetConfig

# Fields that are never treated as sweep axes. n_concepts_list is already a
# list with special meaning within a single run (swept internally).
_FIXED_FIELDS = {"dataset", "n_concepts_list"}

# Fields that belong to DatasetConfig, looked up from the registry.
_DATASET_FIELDS = {"dataset"}


def _canonical_run_config_from_dict(entry: dict) -> dict:
    """Return a JSON-compatible dict with unused sparsity fields removed.

    This is used for deduplication: two RunConfigs that differ only in
    a parameter that is irrelevant given their sparsity_mode are considered
    identical.
    """
    d = copy.deepcopy(entry)
    rc = d["run_config"]
    if rc.get("sparsity_mode") == "l1":
        rc.pop("lambda_kl", None)
        rc.pop("target_p", None)
    else:
        rc.pop("lambda_l1", None)
    return d


def _canonical_run_config(run_cfg: RunConfig, ds_cfg: DatasetConfig) -> dict:
    """Return a JSON-compatible dict with unused sparsity fields removed.

    This is used for deduplication: two RunConfigs that differ only in
    a parameter that is irrelevant given their sparsity_mode are considered
    identical.
    """
    d = _config_to_index_dict(run_cfg, ds_cfg)
    return _canonical_run_config_from_dict(d)


def parse_sweep_config(
    path: Path,
) -> tuple[DatasetConfig, list[RunConfig]]:
    """Parse a sweep TOML and return all (DatasetConfig, RunConfig) combinations.

    Args:
        path: Path to the sweep TOML config file.

    Returns:
        ds_cfg: DatasetConfig looked up from the registry.
        run_cfgs: One RunConfig per cartesian-product combination of all
            list-valued fields.

    """
    with path.open("rb") as f:
        raw = dict(tomllib.load(f))

    # Resolve dataset
    dataset_name = raw.pop("dataset", "")
    if dataset_name not in DATASET_REGISTRY:
        msg = f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY)}"
        raise ValueError(msg)
    ds_cfg, _ = DATASET_REGISTRY[dataset_name]

    # n_concepts_list is fixed (scalar list within each run)
    n_concepts_list = raw.pop("n_concepts_list", [5, 10, 20, 30, 50, 100])

    # Separate sweep axes (list values) from fixed scalars
    known_fields = set(RunConfig.__dataclass_fields__) - {"device"}
    sweep_axes: dict[str, list] = {}
    fixed: dict = {"n_concepts_list": n_concepts_list}

    for key, value in raw.items():
        if key not in known_fields:
            continue  # ignore unknown keys silently
        if isinstance(value, list):
            sweep_axes[key] = value
        else:
            fixed[key] = value

    # Cartesian product over all sweep axes
    if not sweep_axes:
        axis_names: list[str] = []
        combinations: list[tuple] = [()]
    else:
        axis_names = list(sweep_axes.keys())
        combinations = list(itertools.product(*sweep_axes.values()))

    device = get_device()
    seen: list[dict] = []
    run_cfgs: list[RunConfig] = []
    for combo in combinations:
        kwargs = {
            **fixed,
            **dict(zip(axis_names, combo, strict=True)),
        }
        cfg = RunConfig(**kwargs, device=device)
        canonical = _canonical_run_config(cfg, ds_cfg)
        if canonical not in seen:
            seen.append(canonical)
            run_cfgs.append(cfg)

    return ds_cfg, run_cfgs


def _config_to_index_dict(run_cfg: RunConfig, ds_cfg: DatasetConfig) -> dict[str, dict]:
    """Serialise (RunConfig, DatasetConfig) to a JSON-compatible dict.

    device is excluded since it is machine-specific.
    img_shape is converted to a list for JSON compatibility.
    """
    rc = _asdict(run_cfg)
    rc.pop("device")
    dc = _asdict(ds_cfg)
    dc["img_shape"] = list(dc["img_shape"])
    return {"run_config": rc, "dataset_config": dc}


class RunIndex:
    """Manages the sweep index file mapping run IDs to their configs.

    The index is a JSON file at ``index_path`` with structure::

        {
            "run_001": {"run_config": {...}, "dataset_config": {...}},
            "run_002": ...
        }

    Args:
        index_path: Path to the index JSON file. Created if it does not exist.

    """

    def __init__(self, index_path: Path) -> None:
        self._path = index_path
        self._index: dict[str, dict] = {}
        if index_path.exists():
            with index_path.open() as f:
                self._index = json.load(f)

    def find_existing(self, run_cfg: RunConfig, ds_cfg: DatasetConfig) -> str | None:
        """Return the run ID of an identical existing run, or None."""
        target = _canonical_run_config(run_cfg, ds_cfg)
        for run_id, entry in self._index.items():
            if _canonical_run_config_from_dict(entry) == target:
                return run_id
        return None

    def next_run_id(self) -> str:
        """Return the next sequential run ID (e.g. 'run_042')."""
        if not self._index:
            return "run_001"
        last = max(self._index.keys())
        n = int(last.split("_")[1])
        return f"run_{n + 1:03d}"

    def register(self, run_id: str, run_cfg: RunConfig, ds_cfg: DatasetConfig) -> None:
        """Add a new run to the index (does not save to disk)."""
        self._index[run_id] = _canonical_run_config(run_cfg, ds_cfg)

    def save(self) -> None:
        """Persist the index to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as f:
            json.dump(self._index, f, indent=2)

from __future__ import annotations

import numpy as np
import torch
from torchvision.datasets import FashionMNIST, MNIST

from lcblm.utils.data import FlatTensorDataset


def _split_indices(n: int, split: str, seed: int = 42) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    cut = int(0.8 * n)
    if split == "train":
        return perm[:cut]
    return perm[cut:]


def _check_split(split: str) -> None:
    if split not in ("train", "val"):
        msg = f"split must be 'train' or 'val', got {split!r}"
        raise ValueError(msg)


def _load_tv_flat(
    dataset_cls: type,
    root: str,
    split: str,
    n_samples: int | None,
) -> FlatTensorDataset:
    _check_split(split)
    ds = dataset_cls(root=root, train=True, download=True)
    # ds.data is [N, H, W] uint8; normalise to [0, 1] float32 and flatten.
    data = ds.data.float().div(255.0).flatten(1)
    idx = _split_indices(data.shape[0], split)
    data = data[idx]
    if n_samples is not None:
        data = data[:n_samples]
    return FlatTensorDataset(data)


def load_mnist(
    root: str, split: str, n_samples: int | None = None
) -> FlatTensorDataset:
    """Load MNIST train data, 80/20 split. Returns FlatTensorDataset with input_dim=784."""
    return _load_tv_flat(MNIST, root, split, n_samples)


def load_fmnist(
    root: str, split: str, n_samples: int | None = None
) -> FlatTensorDataset:
    """Load FashionMNIST train data, 80/20 split. Returns FlatTensorDataset with input_dim=784."""
    return _load_tv_flat(FashionMNIST, root, split, n_samples)


def load_dsprites(
    path: str,
    split: str,
    n_samples: int | None = None,
) -> tuple[FlatTensorDataset, np.ndarray]:
    """Load dSprites npz. Returns FlatTensorDataset (input_dim=4096) and factors [N, 5].

    The five returned factors are shape, scale, orientation, posX, posY (color excluded
    because it is constant in the standard dSprites release).
    """
    _check_split(split)
    npz = np.load(path, allow_pickle=True)
    imgs = npz["imgs"]  # [N, 64, 64] uint8 {0, 1}
    latents = npz["latents_classes"]  # [N, 6]

    idx = _split_indices(imgs.shape[0], split).numpy()
    imgs_split = imgs[idx]
    factors = latents[idx, 1:]  # drop color factor (index 0, always 0)

    if n_samples is not None:
        imgs_split = imgs_split[:n_samples]
        factors = factors[:n_samples]

    data = torch.from_numpy(imgs_split.reshape(-1, 4096).astype(np.float32))
    return FlatTensorDataset(data), factors

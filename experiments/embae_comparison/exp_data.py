# ruff: noqa: N806

"""Data loading utilities.

Each loader returns (X_train, X_test, y_train, y_test, scaler) where X_* are
already globally normalised (StandardScaler fit on train only) float32 tensors.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from sklearn.datasets import fetch_openml, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from .exp_config import DatasetConfig

# Type alias for the loader signature expected by the CLI and run_experiment.
LoadDataFn = Callable[
    [int],
    tuple[Tensor, Tensor, np.ndarray, np.ndarray, StandardScaler],
]


def load_digits_data(
    n_samples: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Tensor, Tensor, np.ndarray, np.ndarray, StandardScaler]:
    """Load sklearn digits, split, and apply per-feature StandardScaler.

    Args:
        n_samples: Total number of images to sample (stratified by class).
        test_size: Fraction of data held out for testing.
        random_state: Seed for the train/test split.

    Returns:
        X_train, X_test, y_train, y_test, scaler

    """
    digits = load_digits()
    X_full = digits.data.astype(np.float32)
    y_full = digits.target

    if n_samples == -1:
        X_sub, y_sub = X_full, y_full
    else:
        # Stratified subsample
        _, X_sub, _, y_sub = train_test_split(
            X_full,
            y_full,
            test_size=n_samples,
            random_state=random_state,
            stratify=y_full,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y_sub,
        test_size=test_size,
        random_state=random_state,
        stratify=y_sub,
    )

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train).astype(np.float32)
    X_test_norm = scaler.transform(X_test).astype(np.float32)

    return (
        torch.from_numpy(X_train_norm),
        torch.from_numpy(X_test_norm),
        y_train,
        y_test,
        scaler,
    )


def load_mnist_data(
    n_samples: int,
    downsample_factor: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Tensor, Tensor, np.ndarray, np.ndarray, StandardScaler]:
    """Load a stratified subset of MNIST, downsample, and apply StandardScaler.

    Images are downsampled from 28x28 using PIL's LANCZOS filter (area
    averaging), which is the standard choice for shrinking images.

    Args:
        n_samples: Total number of images to sample (stratified by class).
        downsample_factor: Factor by which to reduce each spatial dimension.
            Factor 2 gives 14x14 = 196-dim vectors.
        test_size: Fraction of data held out for testing.
        random_state: Seed for the stratified subsample and train/test split.

    Returns:
        X_train, X_test, y_train, y_test, scaler

    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_full = mnist.data.astype(np.float32)  # (70000, 784)
    y_full = mnist.target.astype(int)

    if n_samples == -1:
        X_sub, y_sub = X_full, y_full
    else:
        # Stratified subsample
        _, X_sub, _, y_sub = train_test_split(
            X_full,
            y_full,
            test_size=n_samples,
            random_state=random_state,
            stratify=y_full,
        )

    # Downsample each image with PIL LANCZOS (area averaging)
    orig_h = orig_w = 28
    new_h = orig_h // downsample_factor
    new_w = orig_w // downsample_factor

    downsampled = np.empty((len(X_sub), new_h * new_w), dtype=np.float32)
    for i, flat in enumerate(X_sub):
        img = Image.fromarray(flat.reshape(orig_h, orig_w))
        img = img.resize((new_w, new_h), resample=Resampling.LANCZOS)
        downsampled[i] = np.asarray(img, dtype=np.float32).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        downsampled,
        y_sub,
        test_size=test_size,
        random_state=random_state,
        stratify=y_sub,
    )

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train).astype(np.float32)
    X_test_norm = scaler.transform(X_test).astype(np.float32)

    return (
        torch.from_numpy(X_train_norm),
        torch.from_numpy(X_test_norm),
        y_train,
        y_test,
        scaler,
    )


def denormalize(x_norm: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Invert StandardScaler normalisation.

    Args:
        x_norm: Normalised samples, shape (N, input_dim).
        scaler: Fitted StandardScaler used during loading.

    Returns:
        Array in original pixel space, same shape as input.

    """
    return scaler.inverse_transform(x_norm)


# ── Dataset registry ──────────────────────────────────────────────────────────
# Maps dataset name (as used in config files) to its (DatasetConfig, loader)
# pair. The loader is a zero-argument callable so the CLI can invoke it without
# knowing anything about dataset-specific parameters.

_MNIST_SIDE = 28
_MNIST_DOWNSAMPLED_SIDE = _MNIST_SIDE // 2  # 14

DATASET_REGISTRY: dict[str, tuple[DatasetConfig, LoadDataFn]] = {
    "digits": (
        DatasetConfig(
            name="digits",
            n_samples=-1,
            input_dim=64,
            img_shape=(8, 8),
            img_vmax=16.0,
        ),
        load_digits_data,
    ),
    "mnist": (
        DatasetConfig(
            name="mnist",
            n_samples=-1,
            input_dim=_MNIST_SIDE**2,
            img_shape=(_MNIST_SIDE, _MNIST_SIDE),
            img_vmax=255.0,
        ),
        functools.partial(load_mnist_data, downsample_factor=1),
    ),
    "mnist_14x14": (
        DatasetConfig(
            name="mnist_14x14",
            n_samples=10_000,
            input_dim=_MNIST_DOWNSAMPLED_SIDE**2,
            img_shape=(_MNIST_DOWNSAMPLED_SIDE, _MNIST_DOWNSAMPLED_SIDE),
            img_vmax=255.0,
        ),
        functools.partial(load_mnist_data, downsample_factor=2),
    ),
}


def preview_downsampling(
    downsample_factor: int = 2,
    n_images: int = 8,
    random_state: int = 42,
    save_path: str = "downsampling_preview.png",
) -> None:
    """Display original vs downsampled MNIST images side by side.

    Each column shows one image; top row is the 28x28 original, bottom row is
    the downsampled version. Useful for sanity-checking the resampling quality.

    Args:
        downsample_factor: Spatial reduction factor (same as in load_mnist_data).
        n_images: Number of example images to show.
        random_state: Seed for reproducible image selection.
        save_path: Path to save the figure. Also displayed if a GUI backend is
            available, otherwise saved only.

    """
    orig_side = 28
    new_side = orig_side // downsample_factor

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_full = mnist.data.astype(np.float32)
    y_full = mnist.target.astype(int)

    rng = np.random.default_rng(random_state)
    idxs = rng.choice(len(X_full), size=n_images, replace=False)

    fig, axes = plt.subplots(
        2,
        n_images,
        figsize=(n_images * 1.5, 3.5),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    for col, idx in enumerate(idxs):
        flat = X_full[idx]
        label = y_full[idx]

        orig = flat.reshape(orig_side, orig_side)
        down = np.asarray(
            Image.fromarray(flat.reshape(orig_side, orig_side)).resize(
                (new_side, new_side),
                resample=Resampling.LANCZOS,
            ),
            dtype=np.float32,
        )

        for row, (img, _side) in enumerate([(orig, orig_side), (down, new_side)]):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray_r", vmin=0, vmax=255)
            ax.axis("off")
            if row == 0:
                ax.set_title(str(label), fontsize=9)

    axes[0, 0].set_ylabel(
        f"{orig_side}x{orig_side}",
        fontsize=9,
        rotation=0,
        labelpad=30,
        va="center",
    )
    axes[0, 0].yaxis.set_visible(True)
    axes[1, 0].set_ylabel(
        f"{new_side}x{new_side}",
        fontsize=9,
        rotation=0,
        labelpad=30,
        va="center",
    )
    axes[1, 0].yaxis.set_visible(True)

    fig.suptitle(
        f"MNIST downsampling preview  (factor={downsample_factor})",
        fontsize=12,
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")
    with contextlib.suppress(Exception):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    preview_downsampling()

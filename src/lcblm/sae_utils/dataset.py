from typing import overload

import torch
from geom_median.torch import compute_geometric_median
from torch import Tensor
from torch.utils.data import Dataset, Subset


class SAEDataset(Dataset[Tensor]):
    """A PyTorch Dataset for training Sparse Autoencoders (SAE) using tensor data."""

    def __init__(self, input_data: Tensor) -> None:
        """Initialize the SAETrainingDataset with the provided data tensor."""
        self.input_data = input_data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    @property
    def num_samples(self) -> int:
        return self.input_data.shape[0]

    @property
    def num_features(self) -> int:
        return self.input_data.shape[-1]

    @overload
    def __getitem__(self, idx: int) -> Tensor: ...
    @overload
    def __getitem__(self, idx: list[int]) -> Tensor: ...
    @overload
    def __getitem__(self, idx: slice) -> Tensor: ...
    @overload
    def __getitem__(self, idx: range) -> Tensor: ...
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.input_data[idx]

        if isinstance(idx, (list, range)):
            return torch.vstack([self[i] for i in idx])

        if isinstance(idx, slice):
            return torch.vstack([self[i] for i in range(*idx.indices(len(self)))])

        msg = "Unsupported index type."
        raise ValueError(msg)


def compute_tied_bias(dataset: SAEDataset, sample_every: int = 15) -> Tensor:
    """Init a tied bias tensor using the geometric median of a subset of the dataset.

    Args:
        dataset: The training dataset containing input tensors.
        sample_every: Interval for sampling the dataset to compute the geometric median.
            Only every `sample_every`-th sample is used to reduce memory usage.

    Returns:
        The geometric median tensor.

    """
    subset = Subset(dataset, indices=range(0, len(dataset), sample_every))
    last_dim = subset[:].shape[-1]
    geom_med: Tensor = compute_geometric_median(subset[:].reshape(-1, last_dim)).median
    return geom_med.to(dataset.input_data.dtype)

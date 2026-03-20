from typing import overload

import torch
from geom_median.torch import compute_geometric_median
from torch import Tensor
from torch.utils.data import Dataset


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
    def __getitem__(self, index: int) -> Tensor: ...
    @overload
    def __getitem__(self, index: list[int]) -> Tensor: ...
    @overload
    def __getitem__(self, index: slice) -> Tensor: ...
    @overload
    def __getitem__(self, index: range) -> Tensor: ...
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.input_data[index]

        if isinstance(index, (list, range)):
            return torch.vstack([self[i] for i in index])

        if isinstance(index, slice):
            return torch.vstack([self[i] for i in range(*index.indices(len(self)))])

        msg = "Unsupported index type."
        raise ValueError(msg)


def compute_tied_bias(data: Tensor, sample_every: int = 15) -> Tensor:
    """Init a tied bias tensor using the geometric median of a subset of the dataset.

    Args:
        data: The training dataset.
        sample_every: Interval for sampling the dataset to compute the geometric median.
            Only every `sample_every`-th sample is used to reduce memory usage.

    Returns:
        The geometric median tensor.

    """
    data = data[::sample_every]
    last_dim = data.shape[-1]
    geom_med: Tensor = compute_geometric_median(data.reshape(-1, last_dim)).median
    return geom_med.to(data.dtype)

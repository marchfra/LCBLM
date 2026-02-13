from typing import NamedTuple

from torch import Tensor
from torch.nn import Module


class NormParams(NamedTuple):
    """A NamedTuple that stores normalization parameters for data preprocessing.

    Attributes:
        mu: The mean values used for normalization.
        std: The standard deviation values used for normalization.

    """

    mu: Tensor
    std: Tensor


class LayerNorm(Module):
    """Custom implementation of layer normalization.

    This module normalizes the input tensor along its last dimension by subtracting the
    mean and dividing by the standard deviation, with an epsilon added for numerical
    stability.

    Args:
        eps: A small value to avoid division by zero.

    """

    def __init__(self, eps: float = 1e-12) -> None:
        """Initialize the object with a specified epsilon value.

        Args:
            eps: A small value to avoid division by zero.

        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, NormParams]:
        """Normalize the input tensor along its last dimension.

        Returns:
            A tuple (normalized_tensor, norm_params), where `norm_params` is a
            NamedTuple containing the mean and standard deviation of the tensor along
            the last dimension.

        """
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + self.eps)
        return x, NormParams(mu, std)

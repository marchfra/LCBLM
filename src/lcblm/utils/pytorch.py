import torch
from torch import Tensor


def clamp_positive(x: Tensor) -> Tensor:
    """Clamp the input tensor to (0, +inf)."""
    dtype = x.dtype
    eps = torch.finfo(dtype).eps
    return torch.clamp(x, min=eps)


def clamp_0_1(x: Tensor) -> Tensor:
    """Clamp the input tensor to (0, 1)."""
    dtype = x.dtype
    eps = torch.finfo(dtype).eps
    return torch.clamp(x, min=eps, max=1 - eps)

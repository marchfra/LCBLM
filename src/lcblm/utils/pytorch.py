import torch
from torch import Tensor


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


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

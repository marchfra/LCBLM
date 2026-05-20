from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset


class FlatTensorDataset(Dataset[Tensor]):
    """Dataset of flat ``(input_dim,)`` tensors.

    The dict-learning baselines and the VAEE/SAE training loops consume this
    shape directly. Image/synthetic data loaders produce instances of this
    class; token-embedding datasets convert via
    ``lcblm.training.data.flatten_token_dataset``.
    """

    def __init__(self, data: Tensor) -> None:
        if data.ndim != 2:  # noqa: PLR2004
            msg = f"FlatTensorDataset expects a 2D tensor, got shape {tuple(data.shape)}."
            raise ValueError(msg)
        self.data = data

    @property
    def input_dim(self) -> int:
        return self.data.shape[1]

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

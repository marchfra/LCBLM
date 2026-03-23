from typing import Protocol

from torch import Tensor


class TensorModule(Protocol):
    def forward(self, x: Tensor, /) -> Tensor: ...

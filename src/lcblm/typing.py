from typing import Protocol, TypeAlias

from torch import Tensor, nn


class TensorModule(Protocol):
    def forward(self, x: Tensor, /) -> Tensor: ...


class TypedLinear(nn.Linear):
    Output: TypeAlias = Tensor

    def forward(self, input: Tensor) -> Output:  # noqa: A002
        return super().forward(input)

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)

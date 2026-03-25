from typing import Protocol, TypeAlias

from torch import Tensor, nn


class TensorModule(Protocol):
    def forward(self, x: Tensor, /) -> Tensor: ...


class ShapedTensorModule(TensorModule, Protocol):
    input_dim: int
    output_dim: int


class TypedLinear(nn.Linear):
    Output: TypeAlias = Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device=None,  # noqa: ANN001
        dtype=None,  # noqa: ANN001
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.input_dim = in_features
        self.output_dim = out_features

    def forward(self, input: Tensor) -> Output:  # noqa: A002
        return super().forward(input)

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)

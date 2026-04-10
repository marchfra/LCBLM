from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from lcblm.typing import TypedLinear

if TYPE_CHECKING:
    from lcblm.typing import TensorModule


class SparseAE(Module):
    """SparseAE is a PyTorch Module implementing a Sparse AutoEncoder (SAE)."""

    class Output(NamedTuple):
        latents: Tensor
        latents_pre_activation: Tensor
        recon: Tensor

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: TensorModule,
        *,
        tied_weights: bool = True,
        use_tied_bias: bool = True,
    ) -> None:
        """Initialize the SparseAE (Sparse AutoEncoder) module.

        Args:
            input_dim: Dimension of the input.
            latent_dim: Size of latent dimension.
            activation: Activation function to use in the network.
            tied_weights: Whether to initialize the weights of the decoder to the
                transpose of the weights of the encoder.
            use_tied_bias: Whether to use the tied bias in the AutoEncoder.

        """
        if not isinstance(activation, nn.Module):
            msg = (
                f"Expected scoring_module to be a torch.nn.Module subclass, "
                f"got {type(activation)}"
            )
            raise TypeError(msg)

        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_tied_bias = use_tied_bias
        self.eps = 1e-7

        # See https://transformer-circuits.pub/2023/monosemantic-features/index.html
        if not self.use_tied_bias:
            self.tied_bias = torch.zeros(self.input_dim)

        self._encoder = TypedLinear(
            in_features=self.input_dim,
            out_features=self.latent_dim,
            bias=True,
        )
        self.activation = activation
        self._decoder = TypedLinear(
            in_features=self.latent_dim,
            out_features=self.input_dim,
            bias=False,
        )

        if tied_weights:
            self._decoder.weight.data = self._encoder.weight.data.T.clone()

    @property
    def device(self) -> torch.device:
        """Get the device of the parameters."""
        return next(self.parameters()).device

    def init_tied_bias(self, tied_bias: Tensor) -> None:
        """Initialize the tied bias parameter.

        Args:
            tied_bias: Tensor to initialize the tied bias.

        """
        if not self.use_tied_bias:
            print("Warning: model not set up to use tied bias")
            return

        if tied_bias.shape != (self.input_dim,):
            msg = (
                f"tied_bias must have shape {(self.input_dim,)}, "
                f"but got {tied_bias.shape}",
            )
            raise ValueError(msg)

        self.tied_bias = nn.Parameter(tied_bias.to(self.device))

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        try:
            _ = self.tied_bias
        except AttributeError as e:
            msg = "tied_bias is not initialized. Call init_tied_bias() first."
            raise AttributeError(msg) from e

        x = x - self.tied_bias
        z_pre_act = self._encoder(x)
        z = self.activation(z_pre_act)
        return z_pre_act, z

    def decode(self, z: Tensor) -> Tensor:
        """Decode the latent representation `z` into the reconstructed input tensor.

        The decoding process involves:
        1. Passing the latent representation `z` through the decoder layer.
        2. Adding the tied bias to the decoded output.

        Args:
            z: Latent representation tensor to be decoded.

        Returns:
            The reconstructed input tensor after decoding.

        """
        return self._decoder(z) + self.tied_bias

    def forward(self, x: Tensor) -> Output:
        """Perform a forward pass through the Sparse AutoEncoder (SAE).

        This method encodes the input to a latent representation, applies the activation
        function, and reconstructs the input from the latent representation.

        Args:
            x: Input tensor to be processed.

        Returns:
            A NamedTuple (`latents`, `latents_pre_activation`, `recon`), where `latents`
            is the activated latent representation, `latents_pre_activation` is the
            latent representation before activation, and `recon` is the reconstructed
            input tensor.

        """
        z_pre_act, z = self.encode(x)
        recon = self.decode(z)

        return self.Output(
            latents=z,
            latents_pre_activation=z_pre_act,
            recon=recon,
        )

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)

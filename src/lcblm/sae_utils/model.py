from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from .normalization import LayerNorm, NormParams


class SAEOutput(NamedTuple):
    """A NamedTuple that stores the results of a Sparse Autoencoder (SAE) forward pass.

    Attributes:
        latents: The encoded latent representations produced by the SAE.
        latents_pre_activation: The latent representations before activation is applied.
        reconstructed_input: The input reconstructed by the SAE decoder.
        norm: Parameters used for input normalization.

    """

    latents: Tensor
    latents_pre_activation: Tensor
    recon: Tensor
    norm: NormParams


class SparseAE(Module):
    """SparseAE is a PyTorch Module implementing a Sparse AutoEncoder (SAE).

    The SAE consists of an encoder and a decoder with tied weights, and includes
    preprocessing steps such as normalization. The latent space is enforced to be sparse
    using a top-k activation function.

    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation: nn.Module,
        epsilon: float = 1e-7,
        *,
        tied_weights: bool = True,
    ) -> None:
        """Initialize the SparseAE (Sparse AutoEncoder) module.

        Args:
            input_dim: Dimension of the input.
            latent_dim: Size of latent dimension.
            activation: Activation function to use in the network.
            epsilon: Small value for numerical stability in normalization.
            tied_weights: Whether to initialize the weights of the decoder to the
                transpose of the weights of the encoder.

        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.eps = epsilon

        self.tied_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.normalization = LayerNorm(eps=epsilon)
        self.lin_encoder = nn.Linear(
            in_features=self.input_dim,
            out_features=self.latent_dim,
            bias=False,
        )
        self.activation = activation
        self.lin_decoder = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.input_dim,
            bias=False,
        )

        if tied_weights:
            self.lin_decoder.weight.data = self.lin_encoder.weight.data.T.clone()

    @property
    def device(self) -> torch.device:
        """Get the device of the parameters."""
        return next(self.parameters()).device

    def init_tied_bias(self, tied_bias: Tensor) -> None:
        """Initialize the tied bias parameter.

        Args:
            tied_bias: Tensor to initialize the tied bias.

        """
        if tied_bias.shape != (self.input_dim,):
            msg = (
                f"tied_bias must have shape ({self.input_dim},), "
                f"but got {tied_bias.shape}",
            )
            raise ValueError(msg)

        self.tied_bias.data = tied_bias.to(self.device)

    def encode_pre_activation(self, x: Tensor) -> Tensor:
        """Compute the pre-activation output of the encoder.

        Before encoding, the tied bias is subtracted from the input tensor.

        Args:
            x: Input tensor to the encoder.

        Returns:
            The encoder's pre-activation output.

        """
        try:
            _ = self.tied_bias
        except AttributeError as e:
            msg = "tied_bias is not initialized. Call init_tied_bias() first."
            raise AttributeError(msg) from e

        x = x - self.tied_bias
        z = self.lin_encoder(x)
        return z

    def decode(self, z: Tensor, norm: NormParams) -> Tensor:
        """Decode the latent representation `z` into the reconstructed input tensor.

        The decoding process involves:
        1. Passing the latent representation `z` through the decoder layer.
        2. Adding the tied bias to the decoded output.
        3. Denormalizing the result using the provided normalization parameters.

        Args:
            z: Latent representation tensor to be decoded.
            norm: NamedTuple containing normalization parameters.

        Returns:
            The reconstructed input tensor after decoding and denormalization.

        """
        x_rec = self.lin_decoder(z) + self.tied_bias
        x_rec = x_rec * (norm.std + self.eps) + norm.mu
        return x_rec

    def forward(self, x: Tensor) -> SAEOutput:
        """Perform a forward pass through the Sparse AutoEncoder (SAE).

        This method normalizes the input, encodes it to a latent representation,
        applies the activation function, and reconstructs the input from the latent
        representation.

        Args:
            x: Input tensor to be processed.

        Returns:
            A NamedTuple (`latents`, `latents_pre_activation`, `recon`, `norm`), where
            `latents` is the activated latent representation, `latents_pre_activation`
            is the latent representation before activation, `recon` is the reconstructed
            input tensor, and `norm` is the normalization parameters used during
            processing.

        """
        x, norm = self.normalization(x)
        z_pre_activation = self.encode_pre_activation(x)
        z = self.activation(z_pre_activation)
        x_reconstructed = self.decode(z, norm)

        return SAEOutput(
            latents=z,
            latents_pre_activation=z_pre_activation,
            recon=x_reconstructed,
            norm=norm,
        )

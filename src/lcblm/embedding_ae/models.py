from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

if TYPE_CHECKING:
    from lcblm.typing import ShapedTensorModule, TensorModule


class MLP(nn.Module):
    """MultiLayer Perceptron module."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize the MultiLayer Perceptron module."""
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.linear1(x)
        x: Tensor = self.activation(x)
        return self.linear2(x)

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)


class EmbeddingAE(nn.Module):
    """Embedding AutoEncoder (AE).

    This class implements an autoencoder that learns a set of prototype embeddings. The
    encoder maps input vectors to embeddings, computes their alignment with prototypes,
    and applies a scoring function to produce scores. The decoder reconstructs the input
    from the embeddings multiplied by the respective score.

    Attributes:
        input_dim: Dimension of the input vectors.
        num_embeddings: Number of prototype embeddings.
        embedding_size: Size of each embedding vector.
        scoring_module: Module to convert alignment values to scores.
        prototypes: Learnable prototype embeddings.
        encoder: Encoder network mapping input to embeddings.
        decoder: Decoder network reconstructing input from scores.

    Note: The responsibility for making the autoencoder sparse lies solely with the
        scoring_module, or with external losses.

    """

    class Output(NamedTuple):
        embeddings: Tensor
        scores: Tensor
        alignments: Tensor
        recon: Tensor

    def __init__(  # noqa: PLR0913
        self,
        num_embeddings: int,
        embedding_size: int,
        encoder: ShapedTensorModule,
        decoder: ShapedTensorModule,
        scoring_module: TensorModule,
        *,
        decode_mode: str = "embeddings",
        normalize: bool = False,
    ) -> None:
        """Initialize an Embedding AutoEncoder.

        Args:
            num_embeddings: The number of embeddings of the AE.
            embedding_size: The size of each embedding.
            encoder: Module mapping input_dim -> num_embeddings * embedding_size.
            decoder: Module mapping num_embeddings * embedding_size -> input_dim.
            scoring_module: The Module to convert cosine alignment between embeddings
                and prototypes to a score.
            decode_mode: How to build the decoder input. "embeddings" decodes from
                score-weighted encoder embeddings, "prototypes" decodes from
                score-weighted prototypes, and "convex" uses the score-interpolated
                combination s * [s * p + (1 - s) * e].
            normalize: If True, L2-normalise encoder embeddings and prototypes before
                computing alignment (cosine similarity instead of dot product). This
                bounds alignments to [-1, 1] and prevents dead prototypes caused by
                scale asymmetry between the embedding cloud and far-away prototypes.

        Raises:
            TypeError: if encoder is not a subclass of torch.nn.Module.
            TypeError: if decoder is not a subclass of torch.nn.Module.
            TypeError: if scoring_module is not a subclass of torch.nn.Module.
            ValueError: if encoder.input_dim doesn't match decoder.output_dim.
            ValueError: if encoder.output_dim doesn't match decoder.input_dim.
            ValueError: if encoder.output_dim doesn't match num_embeddings *
                embedding_size.

        """
        if not isinstance(encoder, nn.Module):
            msg = (
                f"Expected encoder to be a torch.nn.Module subclass, "
                f"got {type(encoder)}"
            )
            raise TypeError(msg)
        if not isinstance(decoder, nn.Module):
            msg = (
                f"Expected decoder to be a torch.nn.Module subclass, "
                f"got {type(decoder)}"
            )
            raise TypeError(msg)
        if not isinstance(scoring_module, nn.Module):
            msg = (
                f"Expected scoring_module to be a torch.nn.Module subclass, "
                f"got {type(scoring_module)}"
            )
            raise TypeError(msg)

        if encoder.input_dim != decoder.output_dim:
            msg = (
                f"Expected encoder.input_dim == decoder.output_dim, "
                f"got {encoder.input_dim} and {decoder.output_dim} respectively"
            )
            raise ValueError(msg)
        if encoder.output_dim != decoder.input_dim:
            msg = (
                f"Expected encoder.output_dim == decoder.input_dim, "
                f"got {encoder.output_dim} and {decoder.input_dim} respectively"
            )
            raise ValueError(msg)
        if encoder.output_dim != num_embeddings * embedding_size:
            msg = (
                f"Expected encoder.output_dim == num_embeddings * embedding_size, "
                f"got {encoder.output_dim} and {num_embeddings * embedding_size} "
                f"respectively"
            )
            raise ValueError(msg)

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self._encoder = encoder
        self._decoder = decoder
        self.scoring_module = scoring_module

        self._prototypes = nn.Parameter(
            torch.randn(self.num_embeddings, self.embedding_size),
        )

        valid_decode_modes = {"embeddings", "prototypes", "convex"}
        if decode_mode not in valid_decode_modes:
            msg = (
                f"Invalid decode_mode {decode_mode!r}. "
                f"Expected one of {sorted(valid_decode_modes)}."
            )
            raise ValueError(msg)

        self._decode_mode = decode_mode
        self._normalize = normalize

    @property
    def device(self) -> torch.device:
        """Get the device of the parameters."""
        return next(self.parameters()).device

    @property
    def prototypes(self) -> Tensor:
        """Get the normalized prototypes of the model."""
        return F.normalize(self._prototypes, dim=-1)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Shape (batch_size, num_embeddings, embedding_size)
        embeddings = self._encoder(x).reshape(
            -1,
            self.num_embeddings,
            self.embedding_size,
        )
        embeddings = F.normalize(embeddings, dim=-1)

        # Shape (batch_size, num_embeddings)
        # Compute alignment between each encoder embedding and its prototype.
        # With normalize=True this is cosine similarity (bounded to [-1, 1]),
        # which prevents dead prototypes caused by scale asymmetry.
        if self._normalize:
            emb_for_align = F.normalize(embeddings, dim=-1)
            proto_for_align = F.normalize(self.prototypes, dim=-1)
        else:
            emb_for_align = embeddings
            proto_for_align = self.prototypes
        alignments = torch.einsum("bne,ne->bn", emb_for_align, proto_for_align)

        # Shape (batch_size, num_embeddings)
        scores: Tensor = self.scoring_module(alignments)

        return embeddings, scores, alignments

    def decode(self, embeddings: Tensor, scores: Tensor) -> Tensor:
        # Scale each embedding by its score and flatten all the embeddings for each
        # sample
        flattened_embeddings = (embeddings * scores.unsqueeze(-1)).flatten(start_dim=1)
        recon = self._decoder(flattened_embeddings)

        return recon

    def forward(self, x: Tensor) -> Output:
        embeddings, scores, alignments = self.encode(x)

        if self._decode_mode == "embeddings":
            decoder_inputs = embeddings
        elif self._decode_mode == "prototypes":
            decoder_inputs = self.prototypes.unsqueeze(0).expand_as(embeddings)
        else:  # self._decode_mode == "convex"
            prototypes = self.prototypes.unsqueeze(0).expand_as(embeddings)
            decoder_inputs = scores.unsqueeze(-1) * prototypes + (
                1 - scores
            ).unsqueeze(-1) * embeddings
            # consider whether to replace scores with a value gamma = 1 - 1/epochs

        recon = self.decode(decoder_inputs, scores)

        return self.Output(
            embeddings=embeddings,
            scores=scores,
            alignments=alignments,
            recon=recon,
        )

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)

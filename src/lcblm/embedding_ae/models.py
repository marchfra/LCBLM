from __future__ import annotations

from typing import NamedTuple, Protocol, TypeAlias

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """MultiLayer Perceptron module."""

    Output: TypeAlias = Tensor

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """Initialize the MultiLayer Perceptron module."""
        super().__init__()

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: Tensor) -> Output:
        x: Tensor = self.linear1(x)
        x: Tensor = self.activation(x)
        return self.linear2(x)

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)


class TensorModule(Protocol):
    def forward(self, x: Tensor, /) -> Tensor: ...


class EmbeddingAE(nn.Module):
    """Embedding AutoEncoder (AE).

    This class implements a autoencoder that learns a set of prototype embeddings. The
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

    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        scoring_module: TensorModule | None = None,
    ) -> None:
        """Initialize an Embedding AutoEncoder.

        Args:
            input_dim: The dimension of the input.
            num_embeddings: The number of embeddings of the AE.
            embedding_size: The size of each embedding.
            scoring_module: The Module to convert cosine alignment between embeddings
                and prototypes to a score. It must implement a forward method that takes
                in a Tensor and returns a Tensor. Defaults to nn.ReLU().

        Raises:
            TypeError: if scoring_module is not a subclass of torch.nn.Module.

        """
        if not isinstance(scoring_module, nn.Module):
            msg = (
                f"Expected scoring_module to be a torch.nn.Module subclass, "
                f"got {type(scoring_module)}"
            )
            raise TypeError(msg)

        super().__init__()

        self.input_dim = input_dim
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.scoring_module = (
            scoring_module if scoring_module is not None else nn.ReLU()
        )

        # NOTE: Explore different initializations
        self.prototypes = nn.Parameter(
            torch.randn(self.num_embeddings, self.embedding_size),
        )
        self._encoder = MLP(self.input_dim, self.num_embeddings * self.embedding_size)
        self._decoder = MLP(self.num_embeddings * self.embedding_size, self.input_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Shape (batch_size, num_embeddings, embedding_size)
        embeddings = self._encoder(x).reshape(
            -1,
            self.num_embeddings,
            self.embedding_size,
        )

        # Shape (batch_size, num_embeddings)
        # This computes the dot product between embedding i and prototype i, for each
        # sample in the batch. The string is an equation that uses Einstein index
        # notation, i.e., sum_{e=0}^{emb_size - 1} embeds_{bne} * prots_{ne} = aligns_{bn}  # noqa: E501
        alignments = torch.einsum("bne,ne->bn", embeddings, self.prototypes)

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

        recon = self.decode(embeddings, scores)

        return self.Output(
            embeddings=embeddings,
            scores=scores,
            alignments=alignments,
            recon=recon,
        )

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)

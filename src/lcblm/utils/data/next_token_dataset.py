from typing import NamedTuple, overload

import torch
from torch import Tensor
from torch.utils.data import Dataset


class Sentence(NamedTuple):
    """Representation of a sentence useful for next token prediction."""

    input_ids: Tensor
    attention_mask: Tensor
    embeddings: Tensor
    next_token_ids: Tensor
    next_attention_mask: Tensor


class NextTokenDataset(Dataset[Sentence]):
    """Dataset useful for next token prediction."""

    def __init__(
        self,
        input_ids: Tensor,
        embeddings: Tensor,
        pad_token_id: int,
        eos_token_id: int,
    ) -> None:
        if input_ids.ndim != 2:  # noqa: PLR2004
            msg = "input_ids must be a 2D tensor."
            raise ValueError(msg)
        if embeddings.ndim != 3:  # noqa: PLR2004
            msg = "embeddings must be a 3D tensor."
            raise ValueError(msg)
        if input_ids.shape != embeddings.shape[:2]:
            msg = "The first two dimensions of input_ids and embeddings must match."
            raise ValueError(msg)

        self.input_ids = input_ids
        self.eos_token_id = eos_token_id
        self.attention_mask = self.input_ids.ne(pad_token_id).int()
        self.embeddings = embeddings

    def __len__(self) -> int:
        return self.num_sentences

    @property
    def num_sentences(self) -> int:
        """The number of sentences in the dataset."""
        return self.embeddings.size(0)

    @property
    def context_window(self) -> int:
        """The number of token in each sentence."""
        return self.embeddings.size(1)

    @property
    def embedding_dimension(self) -> int:
        """The dimension of the LLM's residual flow."""
        return self.embeddings.size(2)

    @overload
    def __getitem__(self, idx: int) -> Sentence: ...
    @overload
    def __getitem__(self, idx: list[int]) -> list[Sentence]: ...
    @overload
    def __getitem__(self, idx: slice) -> list[Sentence]: ...
    def __getitem__(self, idx):
        if isinstance(idx, int):
            input_ids = self.input_ids[idx]
            attention_mask = self.attention_mask[idx]
            embeddings = self.embeddings[idx]

            next_token_ids = torch.cat(
                [input_ids[1:], torch.tensor([self.eos_token_id])],
            )
            next_attention_mask = torch.cat([attention_mask[1:], torch.tensor([1])])

            return Sentence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=embeddings,
                next_token_ids=next_token_ids,
                next_attention_mask=next_attention_mask,
            )

        if isinstance(idx, list):
            return [self[i] for i in idx]

        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        msg = "Unsupported index type."
        raise ValueError(msg)

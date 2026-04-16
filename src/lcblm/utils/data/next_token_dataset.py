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
        attention_mask: Tensor,
        embeddings: Tensor,
        eos_token_id: int,
    ) -> None:
        """Initialize the dataset with input token IDs and their LLM embeddings.

        Args:
            input_ids: A 2D tensor of token IDs with shape (num_sentences, seq_length).
            attention_mask: A 2D tensor that is 0 where a pad token is used and 1
                otherwise.
            embeddings: A 3D tensor of embeddings with shape (num_sentences, seq_length,
                embedding_dim).
            eos_token_id: The token ID used to denote end-of-sequence.

        Raises:
            ValueError: If input_ids is not 2D, embeddings is not 3D, or their first two
                dimensions do not match.

        """
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
        self.attention_mask = attention_mask.bool()
        self.eos_token_id = eos_token_id
        self.embeddings = embeddings.float()

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
    def __getitem__(self, index: int) -> Sentence: ...
    @overload
    def __getitem__(self, index: list[int]) -> list[Sentence]: ...
    @overload
    def __getitem__(self, index: slice) -> list[Sentence]: ...
    @overload
    def __getitem__(self, index: range) -> list[Sentence]: ...
    def __getitem__(self, index):
        if isinstance(index, int):
            input_ids = self.input_ids[index]
            attention_mask = self.attention_mask[index]
            embeddings = self.embeddings[index]

            next_token_ids = torch.cat(
                [input_ids[1:], torch.tensor([self.eos_token_id])],
            )
            # With transformers==5.0.0 the tokenizer pads on the right. This means that,
            # if I use the EOS_TOKEN as padding (which I do), there's no need to shift
            # the attention mask by one
            # If using transformers<5, change the next line to
            # `next_attention_mask = torch.cat([attention_mask[1:], torch.tensor([1])])`
            next_attention_mask = attention_mask

            return Sentence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=embeddings,
                next_token_ids=next_token_ids,
                next_attention_mask=next_attention_mask,
            )

        if isinstance(index, (list, range)):
            return [self[i] for i in index]

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        msg = "Unsupported index type."
        raise ValueError(msg)

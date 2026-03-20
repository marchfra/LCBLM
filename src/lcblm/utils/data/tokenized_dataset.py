import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, overload

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.formatting.formatting import LazyBatch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)

from lcblm._logging import utils_logger as logger

if TYPE_CHECKING:
    from datasets.dataset_dict import DatasetDict


@dataclass
class DatasetConfig:
    """Common configuration for all datasets.

    Args:
        model_id: The id of a HuggingFace pretrained model, used to instantiate the
            tokenizer.
        max_length: The maximum length of a tokenized sequence.
        shuffle_train: Whether to shuffle the train set.
        dataloader_kwargs: Any additional dataloader settings.

    """

    model_id: str
    max_length: int = 512
    shuffle_train: bool = False
    dataloader_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize tokenizer after config creation."""
        os.environ["TOKENIZER_PARALLELISM"] = "false"

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_id,
        )
        if self.tokenizer.pad_token is None:
            existing_special_tokens = list(
                self.tokenizer.special_tokens_map_extended.values(),
            )
            # check that the model already has at least one special token defined
            if len(existing_special_tokens) == 0:
                msg = (
                    "The tokenizer must have at least one special token defined to use "
                    "for padding. Please use a different tokenizer."
                )
                raise ValueError(msg)
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})


class EncodedSentence(NamedTuple):
    """Tokenized sentence representation."""

    input_ids: Tensor
    attention_mask: Tensor


class TokenizedDataset(Dataset[EncodedSentence]):
    """A PyTorch Dataset for handling tokenized sentences and their attention masks.

    This dataset stores input IDs and attention masks as tensors and provides convenient
    access to individual or multiple encoded sentences. It supports integer, list, and
    slice indexing, returning EncodedSentence objects or lists thereof. The dataset also
    exposes properties for the number of sentences and the context window (sequence
    length).

    Args:
        input_ids: Tokenized input IDs for each sentence.
        attention_mask: Attention masks corresponding to input IDs.

    """

    def __init__(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> None:
        """Initialize the dataset with input IDs and attention masks.

        Args:
            input_ids: Tokenized input IDs for each sentence.
            attention_mask: Attention masks corresponding to input IDs.

        """
        if len(input_ids) != len(attention_mask):
            msg = "input_ids and attention_mask must have the same length"
            raise ValueError(msg)

        self.input_ids = torch.tensor(input_ids)
        self.attention_mask = torch.tensor(attention_mask)

    def __len__(self) -> int:
        """Return the number of sentences in the dataset."""
        return self.num_sentences

    @property
    def num_sentences(self) -> int:
        """Get the number of sentences in the dataset."""
        return self.input_ids.size(0)

    @property
    def context_window(self) -> int:
        """Get the context window size (sequence length)."""
        return self.input_ids.size(1)

    @overload
    def __getitem__(self, index: int) -> EncodedSentence: ...
    @overload
    def __getitem__(self, index: list[int]) -> list[EncodedSentence]: ...
    @overload
    def __getitem__(self, index: slice) -> list[EncodedSentence]: ...
    @overload
    def __getitem__(self, index: range) -> list[EncodedSentence]: ...
    def __getitem__(self, index):
        if isinstance(index, int):
            input_ids = self.input_ids[index]
            attention_mask = self.attention_mask[index]

            return EncodedSentence(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if isinstance(index, (list, range)):
            return [self[i] for i in index]

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        msg = "Unsupported index type."
        raise ValueError(msg)


class DatasetStrategy(ABC):
    """Base strategy for dataset loading."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the dataset with the provided configuration.

        Args:
            config: The configuration object containing dataset parameters.

        """
        self.config = config

    @abstractmethod
    def load_datasets(self) -> tuple[HFDataset, HFDataset]:
        """Create train and validation datasets.

        If the validation split of a dataset doesn't exist, use the test split instead.
        """

    @property
    def text_field(self) -> str:
        """The name of the column containing text examples in the HFDataset."""
        return "text"

    def get_loaders(
        self,
        batch_size: int,
    ) -> tuple[DataLoader[EncodedSentence], DataLoader[EncodedSentence]]:
        """Create train and validation dataloaders."""
        train_dataset, val_dataset = self.load_datasets()
        logger.info(
            "Loaded datasets.",
            extra={
                "cls_name": self.__class__.__name__,
                "train_length": len(train_dataset),
                "val_length": len(val_dataset),
            },
        )

        train_tokenized, val_tokenized = self.tokenize_datasets(
            train_dataset,
            val_dataset,
        )
        logger.debug("Tokenized datasets.")

        train_loader = self._create_loader(
            train_tokenized,
            batch_size=batch_size,
            shuffle=self.config.shuffle_train,
        )
        val_loader = self._create_loader(
            val_tokenized,
            batch_size=batch_size,
            shuffle=False,
        )
        logger.debug("Created dataloaders.")

        return train_loader, val_loader

    def tokenize_datasets(
        self,
        train_dataset: HFDataset,
        val_dataset: HFDataset,
    ) -> tuple[TokenizedDataset, TokenizedDataset]:
        """Tokenize a dataset."""
        mapped_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=len(train_dataset),
        )
        mapped_val = val_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=len(val_dataset),
        )

        tokenized_train = TokenizedDataset(
            mapped_train["input_ids"],
            mapped_train["attention_mask"],
        )
        tokenized_val = TokenizedDataset(
            mapped_val["input_ids"],
            mapped_val["attention_mask"],
        )

        return tokenized_train, tokenized_val

    def _create_loader(
        self,
        dataset: TokenizedDataset,
        batch_size: int,
        *,
        shuffle: bool,
    ) -> DataLoader[EncodedSentence]:
        """Create a DataLoader with standard configuration."""
        kwargs = self.config.dataloader_kwargs
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle
        return DataLoader(dataset, **kwargs)

    def tokenize_function(self, examples: LazyBatch) -> BatchEncoding:
        """Tokenize a dataset batch."""
        return self.config.tokenizer(
            examples[self.text_field],
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )


class SST2Strategy(DatasetStrategy):
    """Strategy for SST-2 (Stanford Sentiment Treebank) dataset."""

    def load_datasets(self) -> tuple[HFDataset, HFDataset]:
        """Create train and validation datasets."""
        # Load from HuggingFace
        datasets: DatasetDict = load_dataset("SetFit/sst2")  # type: ignore[reportAssignmentType]

        return datasets["train"], datasets["validation"]


class YelpPolarityStrategy(DatasetStrategy):
    """Strategy for Yelp Polarity dataset."""

    def load_datasets(self) -> tuple[HFDataset, HFDataset]:
        """Create train and validation datasets."""
        datasets: DatasetDict = load_dataset("yelp_polarity")  # type: ignore[reportAssignmentType]

        return datasets["train"], datasets["test"]


class IMDBStrategy(DatasetStrategy):
    """Strategy for IMDB movie reviews dataset."""

    def load_datasets(self) -> tuple[HFDataset, HFDataset]:
        """Create train and validation datasets."""
        datasets: DatasetDict = load_dataset("stanfordnlp/imdb")  # type: ignore[reportAssignmentType]

        return datasets["train"], datasets["test"]


class AGNewsStrategy(DatasetStrategy):
    """Strategy for AG News dataset (4-class news classification)."""

    def load_datasets(self) -> tuple[HFDataset, HFDataset]:
        """Create train and validation datasets."""
        datasets: DatasetDict = load_dataset("ag_news")  # type: ignore[reportAssignmentType]

        return datasets["train"], datasets["test"]


class DatasetFactory:
    """Factory for creating dataset strategies.

    Example usage:
    >>> # with predefined datasets
    >>> dataset_name = "sst2"
    >>> config = DatasetConfig(batch_size=32)
    >>> dataset_strategy = DatasetFactory.create(dataset_name, config)

    >>> # with user-created dataset
    >>> dataset_name = "custom_dataset"
    >>> config = DatasetConfig(batch_size=64)
    >>> DatasetFactory.register("custom_dataset", CustomDatasetStrategy)
    >>> dataset_strategy = DatasetFactory.create(dataset_name, config)

    """

    # This variable is shared between all instances of DatasetFactory
    _strategies: ClassVar[dict[str, type[DatasetStrategy]]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: type[DatasetStrategy]) -> None:
        """Register a new dataset strategy."""
        cls._strategies[name] = strategy_class
        logger.debug(
            "Registered new dataset strategy.",
            extra={
                "new_strategy_name": name,
                "new_strategy_class": strategy_class,
                "available_strategies": cls.available_datasets(),
            },
        )

    @classmethod
    def create(cls, name: str, config: DatasetConfig) -> DatasetStrategy:
        """Create a dataset strategy by name."""
        if name not in cls._strategies:
            msg = (
                f"Unknown dataset: {name}. "
                f"Available datasets: {cls.available_datasets()}"
            )
            raise ValueError(msg)
        return cls._strategies[name](config)

    @classmethod
    def available_datasets(cls) -> list[str]:
        """Get list of available datasets."""
        return list(cls._strategies.keys())


# Register predefined dataset strategies
DatasetFactory.register("sst2", SST2Strategy)
DatasetFactory.register("yelp_polarity", YelpPolarityStrategy)
DatasetFactory.register("imdb", IMDBStrategy)
DatasetFactory.register("ag_news", AGNewsStrategy)

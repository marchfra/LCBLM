from collections.abc import Iterator
from typing import TypeVar

from torch.utils.data import DataLoader

T = TypeVar("T")


def typed_dataloader(dataloader: DataLoader[T]) -> Iterator[T]:
    """Type-safe iterator over torch DataLoader."""
    yield from dataloader

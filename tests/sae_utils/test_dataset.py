import pytest
import torch

from sae_utils.dataset import SAEDataset, compute_tied_bias


def test_sae_training_dataset_init_with_tensor() -> None:
    data = torch.randn(10, 5)
    dataset = SAEDataset(data)
    assert torch.equal(dataset.data, data)
    assert isinstance(dataset.data, torch.Tensor)
    assert dataset.data.shape == (10, 5)


def test_sae_training_dataset_init_with_empty_tensor() -> None:
    data = torch.empty(0, 5)
    dataset = SAEDataset(data)
    assert torch.equal(dataset.data, data)
    assert dataset.data.shape == (0, 5)


def test_sae_training_dataset_len_with_tensor() -> None:
    data = torch.randn(7, 3)
    dataset = SAEDataset(data)
    assert len(dataset) == 7


def test_sae_training_dataset_len_with_empty_tensor() -> None:
    data = torch.empty(0, 4)
    dataset = SAEDataset(data)
    assert len(dataset) == 0


def test_sae_training_dataset_getitem_returns_correct_sample() -> None:
    data = torch.randn(6, 4)
    dataset = SAEDataset(data)
    for idx in range(len(data)):
        sample = dataset[idx]
        assert torch.equal(sample, data[idx])
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (4,)


def test_sae_training_dataset_getitem_with_negative_index() -> None:
    data = torch.randn(5, 2)
    dataset = SAEDataset(data)
    sample = dataset[-1]
    assert torch.equal(sample, data[-1])
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (2,)


def test_sae_training_dataset_getitem_out_of_bounds_raises_index_error() -> None:
    data = torch.randn(3, 2)
    dataset = SAEDataset(data)
    with pytest.raises(IndexError):
        dataset[3]
    with pytest.raises(IndexError):
        dataset[-4]


class DummyMedianResult:
    def __init__(self, median: torch.Tensor) -> None:
        self.median = median


def test_tied_bias_initialization_returns_geometric_median() -> None:
    data = torch.Tensor(
        [
            [5, 5, 5, 5, 5, 5, 5, 5],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    dataset = SAEDataset(data)

    result = compute_tied_bias(dataset, sample_every=5)
    assert torch.allclose(result.cpu(), 5 * torch.ones(8), atol=1e-4)
    assert result.dtype == torch.float32


def test_tied_bias_initialization_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    data = torch.randn(10, 3)
    dataset = SAEDataset(data)

    expected_median = torch.ones(3)
    monkeypatch.setattr(
        "sae_utils.dataset.compute_geometric_median",
        lambda subset: DummyMedianResult(expected_median),
    )
    monkeypatch.setattr(
        torch,
        "cuda",
        type("cuda", (), {"is_available": staticmethod(lambda: True)}),
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    result = compute_tied_bias(dataset)
    assert torch.allclose(result.cpu(), expected_median.float())

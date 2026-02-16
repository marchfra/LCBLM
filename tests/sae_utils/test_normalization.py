import torch

from sae_utils.normalization import LayerNorm


def test_layernorm_default_eps() -> None:
    ln = LayerNorm()
    assert ln.eps == 1e-5


def test_layernorm_custom_eps() -> None:
    custom_eps = 1e-3
    ln = LayerNorm(eps=custom_eps)
    assert ln.eps == custom_eps


def test_layernorm_forward_normalization() -> None:
    ln = LayerNorm()
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
    )
    normed, params = ln.forward(x)
    # Check shape
    assert normed.shape == x.shape
    assert params.mu.shape == (2, 1)
    assert params.std.shape == (2, 1)
    # Check mean is close to 0 and std is close to 1
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normed.std(dim=-1), torch.ones(2), atol=1e-6)


def test_layernorm_forward_eps_effect() -> None:
    x = torch.ones((2, 3))
    ln = LayerNorm()
    normed, _params = ln.forward(x)
    # Since std is zero, output should be zeros
    assert torch.allclose(normed, torch.zeros_like(x))


def test_layernorm_forward_batch_and_dim() -> None:
    ln = LayerNorm()
    x = torch.randn(4, 5, 6)
    normed, params = ln.forward(x)
    assert normed.shape == x.shape
    assert params.mu.shape == (4, 5, 1)
    assert params.std.shape == (4, 5, 1)

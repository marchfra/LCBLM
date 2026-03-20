from typing import TypeAlias

import torch
from torch import Tensor
from torch.nn import Module

from lcblm.utils import clamp_0_1


class TopK(Module):
    def __init__(self, k: int) -> None:
        """Initialize the TopK module.

        Args:
            k: The number of top activations to keep.

        Raises:
            ValueError: if k <= 0.

        """
        if k <= 0:
            msg = f"k should be greater than zero, got {k}"
            raise ValueError(msg)

        super().__init__()
        self.k = k

    def forward(self, z: Tensor) -> Tensor:
        """Apply a top-k selection to the input tensor along the last dimension.

        This method selects the top k largest values from the input tensor z along the
        last dimension, and returns a tensor of the same shape as z where only the top-k
        values are retained at their respective positions, and all other elements are
        set to 0.

        Args:
            z: Input tensor of arbitrary shape.

        Returns:
            Output tensor of the same shape as z, with only the top-k values retained
            along the last dimension and all other elements set to 0.

        Raises:
            ValueError: If k is greater than the size of the last dimension of z.

        """
        if self.k > z.shape[-1]:
            msg = (
                f"k cannot be greater than the last dimension of the input tensor. "
                f"Got k={self.k} and input tensor with last dimension size "
                f"{z.shape[-1]}"
            )
            raise ValueError(msg)

        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        output = torch.zeros_like(z).scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_values,
        )
        return output

    def __str__(self) -> str:
        """Return a string representation of the TopK module."""
        return f"{self.__class__.__name__}(k={self.k})"


def _all_except_last_dim(tensor: Tensor, *, keepdim: bool = False) -> Tensor:
    """Check if all elements are True along all dimensions except the last one.

    Args:
        tensor: Input boolean tensor of arbitrary shape.
        keepdim: Whether to retain reduced dimensions with size 1.

    Returns:
        A boolean tensor containing the result of the torch.all() operation along all
        dimensions except the last one. (shape: [size_of_last_dim] if keepdim is False,
        else shape: [1, 1, ..., size_of_last_dim])

    """
    ndim = tensor.dim()
    dims_to_reduce = tuple(range(ndim - 1))
    return tensor.all(dim=dims_to_reduce, keepdim=keepdim)


def update_dead_latent_counts(activations: Tensor, prev_counts: Tensor) -> Tensor:
    """Update the count of dead latents based on the current batch activations.

    A latent is considered "dead" in a batch if for every token in the context of each
    sample its activation is zero.
    This function increments the dead latent count for each latent that is dead in the
    current batch, and resets the count to zero for latents that are active.

    Example:
        Given the following latent activations `z` for a batch of 2 samples, each with a
        context length of 3 and 5 latents:
        >>> activations = [[[0, 0, 0, 0, 5],
                            [0, 0, 3, 2, 1],
                            [0, 0, 3, 4, 5]],
                           [[0, 2, 0, 4, 5],
                            [0, 4, 3, 2, 1],
                            [0, 2, 3, 4, 5]]]
        >>> is_dead =       [1, 0, 0, 0, 0]

    Args:
        activations: The latent activations tensor. (shape [batch_size, context_len,
            num_latents]).
        prev_counts: The tensor containing previous dead neuron counts for each
            neuron. (shape: [num_latents])

    Returns:
        Updated dead batches counts for each latent.

    """
    # 0 is active, 1 is inactive
    dead_mask = _all_except_last_dim(activations == 0).to(dtype=torch.int)
    count = prev_counts * dead_mask  # This resets the count if the latent is active
    count += dead_mask
    return count


def gumbel_sigmoid(logit_p: Tensor, tau: float = 1.0, *, hard: bool = False) -> Tensor:
    """Sample from independent Bernoulli distributions using the Gumbel-Sigmoid trick.

    Produce a differentiable continuous relaxation of Bernoulli sampling by adding
    Logistic noise (equivalent to Gumbel noise difference) in logit space. This is
    the binary/multilabel analog of the Gumbel-Softmax (Concrete) distribution.

    As temperature approaches 0, samples approach hard binary decisions. As temperature
    increases, samples become more uniform.

    The function supports both soft (continuous) and hard (binary) outputs. When
    hard=True, it uses the Straight-Through Estimator (STE) to produce discrete samples
    in the forward pass while maintaining differentiability in the backward pass.

    Theory:
        Classical Bernoulli sampling with E[x] = p is:
            z = 1[u < p],  where u ~ Uniform(0, 1)

        Using p = sigmoid(l) where l = logit(p), we can shift the computation to u:
            u < 1 / (1 + exp(-l))
            1 + exp(-l) < 1 / u
            exp(-l) < (1 - u) / u
            -l < log((1 - u) / u)
            l > log(u / (1 - u)) = logit(u) = g

        So the sampling becomes:
            z = 1[l > g],  where g = logit(u) ~ Logistic(0, 1)

        Properties of g (Logistic noise):
            - E[g] = 0
            - Symmetric around 0
            - Has a closed-form PDF (see PSF04)

        To make this differentiable, replace the indicator function 1[l > g] with
        a temperature-controlled sigmoid:
            z = sigmoid((l - g) / τ)

        This is the Concrete/Gumbel-Sigmoid relaxation. The noise g can equivalently
        be added or subtracted since it's zero-mean and symmetric; we subtract here
        to match the indicator function formulation l > g.

    Args:
        logit_p: Logits (unnormalized log-odds) for each independent Bernoulli. Can be
            any shape. These are the log(p/(1-p)) values, NOT log probabilities.
        tau: Positive temperature parameter controlling the relaxation. Lower values
            produce sharper (more discrete) samples.
        hard: If True, returns hard binary samples (0 or 1) using the Straight-Through
            Estimator while maintaining gradients. If False, returns soft continuous
            values in [0, 1].

    Returns:
        Bernoulli samples with the same shape as logit_p. If hard=False, returns
        continuous values in [0, 1]. If hard=True, returns binary values {0, 1} with
        gradients flowing through the soft relaxation (STE).

    Raises:
        ValueError: if tau isn't positive.

    Examples:
        >>> # Soft samples during training
        >>> logits = torch.randn(5)  # alignment scores or learned logits
        >>> soft_samples = gumbel_sigmoid(logits, tau=0.5)
        >>>
        >>> # Hard samples with gradients (Straight-Through Estimator)
        >>> hard_samples = gumbel_sigmoid(logits, tau=0.5, hard=True)
        >>>
        >>> # Common pattern: soft during training, hard at inference
        >>> samples = gumbel_sigmoid(logits, tau=0.5, hard=not model.training)

    Note:
        This implements the reparameterization trick for Bernoulli distributions.
        The noise term logit(u) = log(u/(1-u)) for u ~ Uniform(0,1) is equivalent
        to the difference of two Gumbel(0,1) variables, which has a Logistic(0,1)
        distribution. This preserves the mathematical structure of the Gumbel-Max
        trick while staying numerically stable in logit space.

        Also known as: Binary Concrete distribution, Relaxed Bernoulli, or
        Gumbel-Sigmoid relaxation.

    References:
        Maddison et al. "The Concrete Distribution: A Continuous Relaxation of
        Discrete Random Variables." ICLR 2017.

        Jang et al. "Categorical Reparameterization with Gumbel-Softmax." ICLR 2017.

    """
    if tau <= 0:
        msg = "tau must be strictly positive"
        raise ValueError(msg)

    u = torch.rand_like(logit_p)
    u = clamp_0_1(u)  # Avoid log(0) and division by 0 divergence
    logit_u = torch.log(u / (1 - u))
    y_soft = torch.sigmoid((logit_p - logit_u) / tau)

    if not hard:
        return y_soft

    # Straight-Through Estimator
    probability_threshold = 0.5
    y_hard = (y_soft > probability_threshold).float()
    return y_hard + y_soft - y_soft.detach()


class GumbelSigmoid(Module):
    Output: TypeAlias = Tensor

    def __init__(self, tau: float = 1.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, x: Tensor) -> Output:
        return gumbel_sigmoid(x, tau=self.tau, hard=self.training)

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)


def bernoulli_hard_sample(logit_p: Tensor) -> Tensor:
    """Hard Bernoulli samples from logits (deterministic, for inference).

    Args:
        logit_p: Logits for each Bernoulli decision.

    Returns:
        Hard binary samples (0 or 1).

    """
    return (logit_p > 0).float()

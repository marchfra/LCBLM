import torch
from torch import Tensor
from torch.nn.functional import mse_loss

from lcblm.utils import clamp_positive

from .activations import TopK
from .model import SparseAE

loss_recon_fn = mse_loss


def loss_k_aux(
    autoencoder: SparseAE,
    x: Tensor,
    sae_output: SparseAE.Output,
    dead_latents_mask: Tensor,
    k_aux: int = 512,
) -> Tensor:
    """Compute the auxiliary k-sparse loss for a sparse autoencoder.

    This loss measures the mean squared error (MSE) between the reconstruction error and
    its approximation using only the top-k activated dead neurons. It encourages the
    autoencoder to avoid dead neurons by promoting activity in these neurons.
    For more details, see https://cdn.openai.com/papers/sparse-autoencoders.pdf,
    Sections 2.4 and A.2.

    Args:
        autoencoder: The sparse autoencoder model with a decode method.
        x: The original input tensor.
        sae_output: The output of the autoencoder's forward pass, containing:
            - latents: The post-activation latent tensor.
            - latents_pre_activation: The pre-activation latent tensor.
            - recon: The reconstructed input tensor.
            - norm: Normalization parameters used during processing.
        dead_latents_mask: A mask tensor indicating inactive (dead) neurons. A value of
            1 indicates a dead neuron, and 0 indicates an active neuron.
        k_aux: Number of top activations to use for the auxiliary loss. Defaults to 512.

    Returns:
        The computed auxiliary k-sparse loss (MSE), with NaNs replaced by zero.


    """
    topk_aux = TopK(k=k_aux)

    e = x - sae_output.recon
    dead_pre_activations = sae_output.latents_pre_activation * dead_latents_mask
    e_hat = autoencoder.decode(topk_aux(dead_pre_activations))
    return mse_loss(e, e_hat, reduction="mean").nan_to_num(0)


def loss_top_k(
    recon_loss: Tensor,
    aux_loss: Tensor,
    alpha_aux: float = 1 / 32,
) -> Tensor:
    """Compute the combined loss for top-k selection by summing recon and aux losses.

    Args:
        recon_loss: The reconstruction loss tensor.
        aux_loss: The auxiliary loss tensor.
        alpha_aux: Weight factor for the auxiliary loss.

    Returns:
        The combined loss.

    """
    return recon_loss + alpha_aux * aux_loss


def bernoulli_kl_loss_from_logits(logits: Tensor, p_prior: float) -> Tensor:
    """Compute KL divergence between concept activations and prior Bernoulli.

    This loss encourages concept activations to match a target sparsity level by
    measuring the divergence between the empirical Bernoulli distribution (estimated
    from batch statistics) and a prior Bernoulli(prior) distribution. Commonly used
    for enforcing sparsity in concept-based models or sparse autoencoders.

    The KL divergence for Bernoulli distributions is:
        KL(p || q) = (1 - p) * log((1 - p) / (1 - q)) + p * log(p / q)

    where p is the empirical probability (averaged sigmoid of logits over the batch)
    and q is the prior probability p_prior.

    Args:
        logits: Concept logits of shape (batch_size, n_concepts) or (batch_size,
            n_concepts, 1). Each entry represents the unnormalized activation/alignment
            for a concept.
        p_prior: Prior probability parameter for the Bernoulli distribution. Must be in
            the range (0, 1) exclusive.

    Returns:
        Tensor: Scalar KL divergence loss, normalized by sqrt(n_concepts).

    Raises:
        ValueError: If p_prior is not in the range (0, 1) exclusive.
        ValueError: If logits tensor has less than 2 dimensions.

    Examples:
        >>> # Encourage 5% concept activation (sparse)
        >>> logits = torch.randn(32, 100)  # 32 samples, 100 concepts
        >>> loss = bernoulli_kl_loss(logits, p_prior=0.05)
        >>>
        >>> # Encourage balanced activation
        >>> loss = bernoulli_kl_loss(logits, p_prior=0.5)

    Note:
        - The loss is normalized by sqrt(n_concepts) to make it scale-invariant
        - Lower p_prior values encourage sparser concept activations

    Warning:
        The batch dimension (dim=0) is averaged over to compute empirical probabilities.
        Ensure your batch size is large enough for stable statistics (recommended:
        >=16).

    """
    if not (0 < p_prior < 1):
        msg = f"p_prior must be in (0, 1) exclusive, got {p_prior}"
        raise ValueError(msg)

    if logits.dim() < 2:  # noqa: PLR2004
        msg = (
            f"logit must have at least 2 dimensions (batch, concepts), "
            f"got shape {logits.shape}",
        )
        raise ValueError(msg)

    n_concepts = logits.shape[1]

    # Average over batch (dim=0) to get per-concept activation probability
    p = torch.sigmoid(logits).mean(dim=0)  # shape: (n_concepts,) or (n_concepts, 1)

    # NOTE: This is actually the KL(q || p), where q is the approximating distribution
    # and p is the prior distribution. This is a bit weird, since usually we do
    # KL(p || q)
    first_term = (1 - p) * torch.log(clamp_positive(1 - p) / (1 - p_prior))
    second_term = p * torch.log(clamp_positive(p) / p_prior)
    kl_div = (first_term + second_term).sum()

    # This makes the loss magnitude comparable across models with different concept
    # counts
    kl_div /= n_concepts

    return kl_div


def bernoulli_kl_loss_from_probs(probs: Tensor, p_prior: float) -> Tensor:
    """Compute KL divergence from already-computed Bernoulli probabilities.

    Same as :func:`bernoulli_kl_loss` but accepts probabilities in [0, 1] directly
    instead of logits — no internal sigmoid is applied. Use this when the caller
    already has soft scores (e.g. GumbelSigmoid outputs) so that the empirical
    activation rate is computed from the actual scores rather than re-derived from
    pre-activation logits.

    Args:
        probs: Soft activation probabilities of shape (batch_size, n_concepts).
            Each entry must be in [0, 1].
        p_prior: Prior activation probability, must be in (0, 1) exclusive.

    Returns:
        Scalar KL divergence loss, normalised by sqrt(n_concepts).

    """
    if not (0 < p_prior < 1):
        msg = f"p_prior must be in (0, 1) exclusive, got {p_prior}"
        raise ValueError(msg)

    if probs.dim() < 2:  # noqa: PLR2004
        msg = (
            f"probs must have at least 2 dimensions (batch, concepts), "
            f"got shape {probs.shape}",
        )
        raise ValueError(msg)

    n_concepts = probs.shape[1]

    p = probs.mean(dim=0)  # empirical activation rate per concept

    first_term = (1 - p) * torch.log(clamp_positive(1 - p) / (1 - p_prior))
    second_term = p * torch.log(clamp_positive(p) / p_prior)
    kl_div = (first_term + second_term).sum()
    kl_div /= n_concepts

    return kl_div

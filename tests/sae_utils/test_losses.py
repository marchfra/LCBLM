import pytest
import torch
from torch import tensor

from sae_utils.activations import TopK, _all_except_last_dim
from sae_utils.losses import loss_k_aux, loss_top_k
from sae_utils.model import SparseAE


@pytest.mark.parametrize(
    ("recon_loss", "aux_loss", "alpha_aux"),
    [
        (0, 0, 0),
        (3.14, 2.72, 1),
        (3.14, 0, 1),
    ],
)
def test_loss_top_k(recon_loss: float, aux_loss: float, alpha_aux: float) -> None:
    recon_loss_ = tensor(recon_loss)
    aux_loss_ = tensor(aux_loss)

    assert (
        loss_top_k(recon_loss_, aux_loss_, alpha_aux=alpha_aux)
        == recon_loss_ + alpha_aux * aux_loss_
    )


def test_loss_k_aux() -> None:
    topk_activation = TopK(k=2)
    autoencoder = SparseAE(input_dim=9, latent_dim_factor=1, activation=topk_activation)
    autoencoder.lin_encoder.weight.data = 2 * torch.eye(n=9)
    autoencoder.lin_decoder.weight.data = 0.5 * torch.eye(n=9)
    # print(f"{autoencoder.lin_encoder.weight = }")
    # print(f"{autoencoder.lin_decoder.weight = }")
    # print(f"{autoencoder.tied_bias = }")
    # print(f"{autoencoder.normalization = }")
    # print(f"{autoencoder.activation = }")
    x = tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [2, 3, 4, 5, 6, 13, 8, 9, 11],
        ],
        dtype=torch.float32,
    )
    print(f"{x = }")
    sae_output = autoencoder(x)
    print(f"{sae_output = }")
    prev_count = torch.tensor([0, 3, 4, 1, 5, 12, 0, 7, 4])
    print(f"{prev_count = }")
    dead_mask = _all_except_last_dim(sae_output.latents == 0).to(dtype=torch.int)
    print(f"{dead_mask = }")
    count = prev_count * dead_mask
    print(f"{count = }")
    count += dead_mask
    print(f"{count = }")
    threshold = 3
    dead_latents_mask = count > threshold
    print(f"{dead_latents_mask = }")
    k_aux = 4
    topk_aux = TopK(k=k_aux)

    e = x - sae_output.recon
    print(f"{e = }")
    dead_pre_activations = sae_output.latents_pre_activation * dead_latents_mask
    print(f"{dead_pre_activations = }")
    dead_post_activations = topk_aux(dead_pre_activations)
    print(f"{dead_post_activations = }")
    e_hat = autoencoder.decode(dead_post_activations, sae_output.norm)
    print(f"{e_hat = }")

    aux_loss = loss_k_aux(autoencoder, x, sae_output, dead_latents_mask, k_aux)
    print(f"{aux_loss = }")

    assert False

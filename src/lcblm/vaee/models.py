from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.functional import cosine_similarity, mse_loss

from lcblm.embedding_ae.models import MLP
from lcblm.utils import clamp_0_1, clamp_positive


class VAEE(nn.Module):
    class Output(NamedTuple):
        recon: Tensor
        mu: Tensor
        alpha: Tensor
        c: Tensor

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        num_embeddings: int = 16,
        embedding_size: int = 16,
        gumbel_temp: float = 0.5,
        output_activation: nn.Module | None = None,
        *,
        encoder_type: Literal["mlp", "linear", "shallow"] = "mlp",
        sigma_0: float = 1.0,
        sim_metric: Literal["cosine", "inner_product", "neg_euclidean"] = "cosine",
        topology: Literal["stacked", "summed"] = "stacked",
    ) -> None:
        if num_embeddings <= 0:
            msg = "num_embeddings must be non-negative."
            raise ValueError(msg)
        if embedding_size <= 0:
            msg = "embedding_size must be non-negative."
            raise ValueError(msg)
        if gumbel_temp <= 0:
            msg = "gumbel_temp must be non-negative."
            raise ValueError(msg)
        if sigma_0 < 0:
            msg = "sigma_0 must be non-negative."
            raise ValueError(msg)

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.gumbel_temp = gumbel_temp
        self.sigma_0 = sigma_0
        self.sim_metric = sim_metric
        self.topology = topology

        self._output_activation = (
            output_activation if output_activation is not None else nn.Identity()
        )

        enc_out = self.num_embeddings * self.embedding_size
        # For summed topology the decoder receives a single embedding_size-dim vector,
        # not num_embeddings*embedding_size
        dec_in = enc_out if topology == "stacked" else self.embedding_size

        match encoder_type:
            case "mlp":
                self._encoder = MLP(input_dim, hidden_dim, enc_out)
                self._decoder = nn.Sequential(
                    MLP(dec_in, hidden_dim, input_dim),
                    self._output_activation,
                )
            case "linear":
                self._encoder = nn.Linear(input_dim, enc_out)
                self._decoder = nn.Sequential(
                    nn.Linear(dec_in, input_dim),
                    self._output_activation,
                )
            case "shallow":
                self._encoder = nn.Sequential(nn.Linear(input_dim, enc_out), nn.GELU())
                self._decoder = nn.Sequential(
                    nn.Linear(dec_in, input_dim),
                    self._output_activation,
                )
            case _:
                msg = (
                    f"Unknown encoder_type: {encoder_type!r}. "
                    f"Must be 'mlp', 'linear', or 'shallow'."
                )
                raise ValueError(msg)

        self.prototypes = nn.Parameter(
            torch.randn(self.num_embeddings, self.embedding_size),
        )

        # Learnable inverse temperature (like OpenAI CLIP).
        # Initializes at 10 which corresponds to tau = 0.1.
        self._logit_scale = nn.Parameter(torch.log(torch.tensor(10.0)))

        # Learnable radius offset b for neg_euclidean metric only.
        # Initialised to E[||mu - prototype||] ≈ sqrt(2*embedding_size) so logits start
        # near 0.
        if sim_metric == "neg_euclidean":
            self._neg_euc_bias = nn.Parameter(
                torch.full((1,), math.sqrt(2.0 * embedding_size)),
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _compute_logits(self, mu: Tensor) -> Tensor:
        """Compute per-concept logits from encoder means and prototypes."""
        scale = torch.clamp(self._logit_scale.exp(), min=1.0, max=100.0)
        match self.sim_metric:
            case "cosine":
                sim = cosine_similarity(mu, self.prototypes.unsqueeze(0), dim=-1)
                return sim * scale
            case "inner_product":
                sim = torch.einsum("bke,ke->bk", mu, self.prototypes)
                return sim * scale
            case "neg_euclidean":
                dist = torch.norm(mu - self.prototypes.unsqueeze(0), dim=-1)
                return (self._neg_euc_bias - dist) * scale
            case _:
                msg = f"Unknown sim_metric: {self.sim_metric!r}"
                raise ValueError(msg)

    def encode(self, x: Tensor) -> Tensor:
        mu = self._encoder(x)
        return mu.reshape(-1, self.num_embeddings, self.embedding_size)

    def sample(
        self,
        mu: Tensor,
        logits: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample c and z from the approximate posterior.

        During training, c uses the Gumbel-Sigmoid relaxation (soft, no STE) and
        z = mu + sigma_0 * eps (stochastic).
        During eval, c = alpha (continuous [0,1]) and z = mu (deterministic).
        """
        if self.training:
            # Gumbel-Sigmoid relaxation (soft samples, no STE)
            u1 = clamp_positive(torch.rand_like(logits))
            u2 = clamp_positive(torch.rand_like(logits))
            logistic_noise = torch.log(u1) - torch.log(u2)
            c = torch.sigmoid((logits + logistic_noise) / self.gumbel_temp)

            # Stochastic z
            eps = torch.randn_like(mu)
            z = mu + self.sigma_0 * eps
        else:
            # Deterministic at eval
            c = alpha
            z = mu

        # Gate z by c
        z = c.unsqueeze(-1) * z

        return z, c

    def decode(self, z: Tensor) -> Tensor:
        """Decode the latent tensor.

        Args:
            z: Shape (batch_size, num_embeddings, embedding_size).

        Returns:
            Reconstructed output of shape (batch_size, input_dim).

        """
        if self.topology == "summed":  # noqa: SIM108
            z_in = z.sum(dim=1)  # (batch_size, embedding_size)
        else:
            z_in = z.flatten(
                start_dim=1,
            )  # (batch_size, num_embeddings * embedding_size)
        return self._decoder(z_in)

    def decoder_first_weight(self) -> Tensor:
        """Return the weight of the first linear layer in the decoder.

        Only meaningful for the stacked topology, where the weight has shape (out_dim,
        num_embeddings * embedding_size) and can be partitioned into num_embeddings
        concept blocks.
        """
        first = self._decoder[0]
        if isinstance(first, MLP):
            return first.linear1.weight
        return first.weight

    def forward(self, x: Tensor) -> Output:
        mu = self._encoder(x).reshape(-1, self.num_embeddings, self.embedding_size)
        logits = self._compute_logits(mu)
        alpha = torch.sigmoid(logits)
        z, c = self.sample(mu, logits, alpha)
        recon = self.decode(z)
        return self.Output(recon=recon, mu=mu, alpha=alpha, c=c)

    def __call__(self, x: Tensor) -> Output:
        return super().__call__(x)


VariationalAutoEmbeddingEncoder = VAEE


class LossOutput(NamedTuple):
    total_loss: Tensor
    recon_loss: Tensor
    cond_kl_loss: Tensor
    sparsity_loss: Tensor
    entropy_loss: Tensor
    ortho_loss: Tensor


def compute_decoder_ortho_loss(
    weight: Tensor,
    num_embeddings: int,
    embedding_size: int,
) -> Tensor:
    """Frobenius orthogonality penalty across decoder concept blocks.

    Partitions the decoder weight matrix into num_embeddings blocks of width
    embedding_size and penalises normalised cross-block inner products:
        sum_{i != j} ||Wi^T Wj||_F^2 / (||Wi||_F^2 * ||Wj||_F^2)

    Args:
        weight: First linear layer weights, shape (out_dim, num_embeddings *
            embedding_size).
        num_embeddings: Number of concept slots.
        embedding_size: Dimension of each concept embedding.

    Returns:
        Scalar penalty tensor.

    """
    blocks = weight.split(
        embedding_size,
        dim=1,
    )  # num_embeddings tensors of shape (out_dim, embedding_size)
    loss = weight.new_zeros(1)
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            cross = (blocks[i].T @ blocks[j]).norm(p="fro") ** 2
            denom = (blocks[i].norm(p="fro") ** 2) * (blocks[j].norm(p="fro") ** 2)
            loss = loss + cross / denom.clamp(min=1e-8)
    return loss


def compute_loss(  # noqa: PLR0913
    target: Tensor,
    input: Tensor,  # noqa: A002
    mu: Tensor,
    alpha: Tensor,
    prototypes: Tensor,
    pi: float,
    gamma: float,
    beta: float,
    lambda_ent: float,
    lambda_ortho: float = 0.0,
    decoder_weight: Tensor | None = None,
    num_embeddings: int = 0,
    embedding_size: int = 0,
) -> LossOutput:
    """Compute the full loss for VAEE training.

    Args:
        target: The original samples to reconstruct.
        input: The reconstruction of the VAEE model.
        mu: The output of the VAEE encoder, shape (batch_size, num_embeddings,
            embedding_size).
        alpha: Bernoulli activation probabilities, shape (batch_size, num_embeddings).
        prototypes: The VAEE's prototypes, shape (num_embeddings, embedding_size).
        pi: Prior Bernoulli activation probability.
        gamma: Coefficient for the conditional KL loss.
        beta: Coefficient for the sparsity KL loss.
        lambda_ent: Coefficient for the entropy regularisation.
        lambda_ortho: Coefficient for the decoder orthogonality penalty.
            Pass 0.0 (default) to disable.
        decoder_weight: First linear layer weight of the decoder, shape
            (out_dim, num_embeddings * embedding_size). Required when lambda_ortho > 0
            and topology is stacked. Pass None to skip the penalty.
        num_embeddings: Required when lambda_ortho > 0.
        embedding_size: Required when lambda_ortho > 0.

    Returns:
        LossOutput with total_loss and all individual terms (unscaled).

    """
    recon_loss = mse_loss(input, target)

    mu_dist = torch.sum((mu - prototypes.unsqueeze(0)) ** 2, dim=-1)
    mu_norm = torch.sum(mu**2, dim=-1)

    alpha_sg = alpha.detach()
    cond_kl = alpha_sg * mu_dist + (1 - alpha_sg) * mu_norm
    cond_kl_loss = cond_kl.sum(dim=-1).mean()

    mean_alpha = alpha.mean(dim=0)
    mean_alpha = clamp_0_1(mean_alpha)
    term_1 = mean_alpha * torch.log((mean_alpha) / pi)
    term_2 = (1 - mean_alpha) * torch.log((1 - mean_alpha) / (1 - pi))
    sparsity_kl = term_1 + term_2
    sparsity_loss = sparsity_kl.sum()

    term_1 = -alpha * torch.log(alpha + 1e-8)
    term_2 = -(1 - alpha) * torch.log(1 - alpha + 1e-8)
    entropy = term_1 + term_2
    entropy_loss = entropy.sum(dim=-1).mean()

    if lambda_ortho > 0 and decoder_weight is not None:
        ortho_loss = compute_decoder_ortho_loss(
            decoder_weight,
            num_embeddings,
            embedding_size,
        )
    else:
        ortho_loss = target.new_zeros(1)

    total_loss = (
        recon_loss
        + gamma * cond_kl_loss
        + beta * sparsity_loss
        + lambda_ent * entropy_loss
        + lambda_ortho * ortho_loss
    )

    return LossOutput(
        total_loss,
        recon_loss,
        cond_kl_loss,
        sparsity_loss,
        entropy_loss,
        ortho_loss,
    )


# Test training loop
if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms  # ty:ignore[unresolved-import]
    from tqdm import trange

    torch.manual_seed(42)

    def get_mnist_dataloaders(
        batch_size: int = 128,
        validation_split: float = 0.1,
    ) -> tuple[DataLoader, DataLoader]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Automatically scales to [0, 1]
            ],
        )
        full_train_dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

        # Split training data into training and validation sets
        num_train = len(full_train_dataset)
        num_val = int(num_train * validation_split)
        num_train = num_train - num_val

        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [num_train, num_val],
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        return train_loader, val_loader

    def visualize_concepts(
        model: VAEE,
        epoch: int,
        save_dir: str | Path = Path("outputs_mnist"),
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        num_embeddings = model.num_embeddings
        embedding_size = model.embedding_size

        fig, axes = plt.subplots(1, num_embeddings, figsize=(num_embeddings * 1.5, 2))
        if num_embeddings == 1:
            axes = [axes]

        for k in range(num_embeddings):
            z = torch.zeros(1, num_embeddings, embedding_size)
            z[0, k] = model.prototypes[k]
            concept_img = model.decode(z).reshape(28, 28).cpu().detach().numpy()

            axes[k].imshow(concept_img, cmap="gray")
            axes[k].axis("off")
            axes[k].set_title(f"C{k}")

        fig.suptitle(f"Concept Dictionary (Epoch {epoch})")
        fig.tight_layout()
        fig.savefig(save_dir / f"concepts_epoch_{epoch:02d}.png")
        plt.close()

    def visualize_reconstructions(
        x: Tensor,
        x_hat: Tensor,
        epoch: int,
        num_samples: int = 8,
        save_dir: str | Path = Path("outputs_mnist"),
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        num_samples = min(num_samples, x.size(0))
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 1.5, 3))

        for i in range(num_samples):
            ax_orig = axes[0, i] if num_samples > 1 else axes[0]
            ax_orig.imshow(x[i].view(28, 28).cpu().numpy(), cmap="gray")
            ax_orig.axis("off")
            if i == 0:
                ax_orig.set_title("Original")

            ax_recon = axes[1, i] if num_samples > 1 else axes[1]
            ax_recon.imshow(x_hat[i].view(28, 28).cpu().detach().numpy(), cmap="gray")
            ax_recon.axis("off")
            if i == 0:
                ax_recon.set_title("Recon")

        fig.tight_layout()
        fig.savefig(save_dir / f"recon_epoch_{epoch:02d}.png")
        plt.close()

    def plot_learning_curves(
        train_losses: list[float],
        val_losses: list[float],
        metric_name: str = "Reconstruction Loss",
        save_dir: str | Path = Path("outputs_mnist"),
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Training", linestyle="-")
        plt.plot(epochs, val_losses, label="Validation", linestyle="-")
        plt.title("Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "learning_curves.png")
        plt.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataloader, val_dataloader = get_mnist_dataloaders(batch_size=128)
    model = VAEE(
        input_dim=784,
        hidden_dim=256,
        num_embeddings=10,
        embedding_size=16,
        gumbel_temp=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = 50
    warmup_epochs = 10  # Gradually introduce constraints so recon learns first

    train_recon_losses = []
    val_recon_losses = []

    pbar = trange(epochs, desc="Starting training...", unit="epoch")

    for epoch in pbar:
        model.train()
        total_loss_val = 0
        recon_loss_val = 0
        mean_active_concepts = 0
        alpha_means, alpha_mins, alpha_maxs = [], [], []

        # Linear warmup
        warmup_factor = min(1.0, epoch / warmup_epochs)
        cur_beta = 1.0 * warmup_factor
        cur_gamma = 0.01 * warmup_factor
        cur_ent = 0.01 * warmup_factor

        for x, _ in train_dataloader:
            x = x.view(x.size(0), -1).to(device)  # noqa: PLW2901
            optimizer.zero_grad()

            x_hat, mu, alpha, c = model(x)

            loss, recon, cond_kl, sparsity, entropy, ortho = compute_loss(
                x,
                x_hat,
                mu,
                alpha,
                model.prototypes,
                pi=0.1,
                gamma=cur_gamma,
                beta=cur_beta,
                lambda_ent=cur_ent,
            )

            loss.backward()
            optimizer.step()

            total_loss_val += loss.item()
            recon_loss_val += recon.item()
            mean_active_concepts += c.sum(dim=-1).mean().item()

            alpha_means.append(alpha.mean().item())
            alpha_mins.append(alpha.min().item())
            alpha_maxs.append(alpha.max().item())

        avg_train_loss = total_loss_val / len(train_dataloader)
        avg_train_recon_loss = recon_loss_val / len(train_dataloader)
        train_recon_losses.append(avg_train_recon_loss)
        avg_c = mean_active_concepts / len(train_dataloader)

        # ---- Validation Phase ----
        model.eval()
        val_total_loss_val = 0
        val_recon_loss_val = 0
        with torch.no_grad():
            for x_val, _ in val_dataloader:
                x_val = x_val.view(x_val.size(0), -1).to(device)  # noqa: PLW2901
                x_hat_val, mu_val, alpha_val, c_val = model(x_val)
                val_loss, val_recon, *_ = compute_loss(
                    x_val,
                    x_hat_val,
                    mu_val,
                    alpha_val,
                    model.prototypes,
                    pi=0.1,
                    gamma=cur_gamma,
                    beta=cur_beta,
                    lambda_ent=cur_ent,
                )
                val_total_loss_val += val_loss.item()
                val_recon_loss_val += val_recon.item()

        avg_val_loss = val_total_loss_val / len(val_dataloader)
        avg_val_recon_loss = val_recon_loss_val / len(val_dataloader)
        val_recon_losses.append(avg_val_recon_loss)

        # ---- Epoch statistics for pbar ----
        alpha_mean = float(np.mean(alpha_means))
        alpha_min = float(np.min(alpha_mins))
        alpha_max = float(np.max(alpha_maxs))
        tau_scale = model._logit_scale.exp().item()  # noqa: SLF001

        pbar.set_description(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Val Loss {avg_val_loss:.4f} | "
            f"Train Recon {avg_train_recon_loss:.4f} | "
            f"Val Recon {avg_val_recon_loss:.4f} | "
            f"Active {avg_c:.1f}/{model.num_embeddings}",
        )
        pbar.set_postfix(
            {
                "α_mean": f"{alpha_mean:.3f}",  # noqa: RUF001
                "α_min": f"{alpha_min:.3f}",  # noqa: RUF001
                "α_max": f"{alpha_max:.3f}",  # noqa: RUF001
                "τ": f"1/{tau_scale:.2f}",
            },
        )

        # ---- Visualization every 5 epochs ----
        if (epoch + 1) % 5 == 0:
            model.eval()  # Ensure model is in eval mode for consistent visualizations
            with torch.no_grad():
                # Use the last batch from training for visualization
                visualize_reconstructions(x, x_hat, epoch + 1, save_dir="my_model")
                visualize_concepts(model, epoch + 1, save_dir="my_model")

    # Plot learning curves after training
    plot_learning_curves(train_recon_losses, val_recon_losses, save_dir="my_model")

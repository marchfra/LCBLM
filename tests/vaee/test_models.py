import torch
import pytest

from lcblm.vaee.models import VAEE, compute_loss, compute_decoder_ortho_loss


# ── Fixtures ──────────────────────────────────────────────────────────────────

INPUT_DIM = 32
HIDDEN_DIM = 16
NUM_EMB = 4
EMB_SIZE = 8
BATCH = 6


def _make_vaee(**kwargs) -> VAEE:
    defaults = dict(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_embeddings=NUM_EMB,
        embedding_size=EMB_SIZE,
        gumbel_temp=0.5,
    )
    defaults.update(kwargs)
    return VAEE(**defaults)


# ── sigma_0 tests ─────────────────────────────────────────────────────────────


def test_sigma_0_always_samples_train():
    """z should differ from c*mu when sigma_0 > 0 in training mode."""
    torch.manual_seed(0)
    model = _make_vaee(sigma_0=1.0)
    model.train()
    x = torch.randn(BATCH, INPUT_DIM)
    mu = model.encode(x)
    logits = model._compute_logits(mu)
    alpha = torch.sigmoid(logits)
    z, c = model.sample(mu, logits, alpha)
    z_det = c.unsqueeze(-1) * mu
    assert not torch.allclose(z, z_det), "z should be stochastic during training"


def test_sigma_0_no_effect_at_eval():
    """At eval, sigma_0 is ignored: z = c * mu exactly, regardless of sigma_0 value."""
    torch.manual_seed(0)
    model = _make_vaee(sigma_0=1.0)
    model.eval()
    x = torch.randn(BATCH, INPUT_DIM)
    mu = model.encode(x)
    logits = model._compute_logits(mu)
    alpha = torch.sigmoid(logits)
    with torch.no_grad():
        z, c = model.sample(mu, logits, alpha)
    assert torch.allclose(z, c.unsqueeze(-1) * mu), "z must equal c*mu at eval (sigma_0 is training-only)"


def test_sigma_0_zero_no_noise():
    """With sigma_0=0, z must equal c*mu exactly."""
    torch.manual_seed(0)
    for training in (True, False):
        model = _make_vaee(sigma_0=0.0)
        model.train(training)
        x = torch.randn(BATCH, INPUT_DIM)
        mu = model.encode(x)
        logits = model._compute_logits(mu)
        alpha = torch.sigmoid(logits)
        with torch.no_grad():
            z, c = model.sample(mu, logits, alpha)
        expected = c.unsqueeze(-1) * mu
        assert torch.allclose(z, expected), f"z must equal c*mu when sigma_0=0 (training={training})"


# ── c eval sampling tests ─────────────────────────────────────────────────────


def test_c_eval_equals_alpha():
    """At eval, c = sigmoid(logits) (soft, continuous) and equals alpha exactly."""
    model = _make_vaee()
    model.eval()
    x = torch.randn(BATCH, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert (out.c >= 0).all() and (out.c <= 1).all(), "c must be in [0, 1]"
    assert torch.allclose(out.c, out.alpha), "c must equal alpha at eval"


def test_c_train_is_soft_gumbel():
    """At train, c is a soft Gumbel-Sigmoid sample: continuous in (0,1) and stochastic."""
    model = _make_vaee()
    model.train()
    x = torch.randn(BATCH, INPUT_DIM)
    mu = model.encode(x)
    logits = model._compute_logits(mu)
    alpha = torch.sigmoid(logits)

    torch.manual_seed(0)
    _, c1 = model.sample(mu, logits, alpha)
    torch.manual_seed(1)
    _, c2 = model.sample(mu, logits, alpha)

    assert (c1 >= 0).all() and (c1 <= 1).all(), "c must be in [0, 1]"
    assert not torch.allclose(c1, c2), "c must be stochastic at train (Gumbel-Sigmoid noise)"


# ── Similarity metric tests ───────────────────────────────────────────────────


@pytest.mark.parametrize("sim_metric", ["cosine", "inner_product", "neg_euclidean"])
def test_sim_metric_forward_shape(sim_metric):
    """Forward pass produces correct output shapes for all similarity metrics."""
    model = _make_vaee(sim_metric=sim_metric)
    model.eval()
    x = torch.randn(BATCH, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.recon.shape == (BATCH, INPUT_DIM)
    assert out.mu.shape == (BATCH, NUM_EMB, EMB_SIZE)
    assert out.alpha.shape == (BATCH, NUM_EMB)
    assert out.c.shape == (BATCH, NUM_EMB)


def test_neg_euclidean_has_bias_param():
    """neg_euclidean model must have the learnable _neg_euc_bias parameter."""
    model = _make_vaee(sim_metric="neg_euclidean")
    assert hasattr(model, "_neg_euc_bias"), "missing _neg_euc_bias parameter"
    assert isinstance(model._neg_euc_bias, torch.nn.Parameter)


def test_cosine_no_bias_param():
    """cosine / inner_product models must NOT have _neg_euc_bias."""
    for sim_metric in ("cosine", "inner_product"):
        model = _make_vaee(sim_metric=sim_metric)
        assert not hasattr(model, "_neg_euc_bias"), (
            f"unexpected _neg_euc_bias on {sim_metric} model"
        )


@pytest.mark.parametrize("sim_metric", ["cosine", "inner_product", "neg_euclidean"])
def test_sim_metric_alpha_in_range(sim_metric):
    """alpha must always be in (0, 1) for all similarity metrics."""
    model = _make_vaee(sim_metric=sim_metric)
    model.eval()
    x = torch.randn(BATCH, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert (out.alpha >= 0).all() and (out.alpha <= 1).all()


# ── Topology tests ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("topology", ["stacked", "summed"])
def test_topology_output_shape(topology):
    """Both topologies must reconstruct to the correct input shape."""
    model = _make_vaee(topology=topology)
    model.eval()
    x = torch.randn(BATCH, INPUT_DIM)
    with torch.no_grad():
        out = model(x)
    assert out.recon.shape == (BATCH, INPUT_DIM)


def test_topology_stacked_decoder_input_dim():
    """Stacked decoder first weight input dim must equal K * embedding_size."""
    model = _make_vaee(topology="stacked", encoder_type="linear")
    w = model.decoder_first_weight()
    assert w.shape[1] == NUM_EMB * EMB_SIZE


def test_topology_summed_decoder_input_dim():
    """Summed decoder first weight input dim must equal embedding_size only."""
    model = _make_vaee(topology="summed", encoder_type="linear")
    w = model.decoder_first_weight()
    assert w.shape[1] == EMB_SIZE


# ── Decoder orthogonality loss tests ─────────────────────────────────────────


def test_decoder_ortho_loss_orthogonal_blocks():
    """Orthogonal blocks (zero cross-product) should yield loss == 0."""
    K, d, out = 2, 3, 5
    # Two blocks with orthogonal column spaces
    W1 = torch.zeros(out, d)
    W1[0, 0] = 1.0
    W2 = torch.zeros(out, d)
    W2[1, 0] = 1.0
    weight = torch.cat([W1, W2], dim=1)
    loss = compute_decoder_ortho_loss(weight, num_embeddings=K, embedding_size=d)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_decoder_ortho_loss_identical_blocks():
    """Identical blocks should yield loss > 0."""
    K, d, out = 2, 3, 5
    W = torch.randn(out, d)
    weight = torch.cat([W, W], dim=1)
    loss = compute_decoder_ortho_loss(weight, num_embeddings=K, embedding_size=d)
    assert loss.item() > 0.0


def test_decoder_ortho_loss_shape():
    """compute_decoder_ortho_loss must return a scalar tensor."""
    K, d, out = 4, 8, 16
    weight = torch.randn(out, K * d)
    loss = compute_decoder_ortho_loss(weight, num_embeddings=K, embedding_size=d)
    assert loss.shape == (1,) or loss.ndim == 0


# ── compute_loss integration tests ───────────────────────────────────────────


def _loss_inputs(batch: int = BATCH, k: int = NUM_EMB, d: int = EMB_SIZE, dim: int = INPUT_DIM):
    target = torch.randn(batch, dim)
    recon = torch.randn(batch, dim)
    mu = torch.randn(batch, k, d)
    alpha = torch.sigmoid(torch.randn(batch, k))
    prototypes = torch.randn(k, d)
    return target, recon, mu, alpha, prototypes


def test_compute_loss_no_ortho():
    """compute_loss works correctly when lambda_ortho=0 (default)."""
    target, recon, mu, alpha, proto = _loss_inputs()
    out = compute_loss(target, recon, mu, alpha, proto, pi=0.1, gamma=0.01, beta=1.0, lambda_ent=0.01)
    assert out.total_loss.isfinite()
    assert out.ortho_loss.item() == pytest.approx(0.0, abs=1e-9)


def test_compute_loss_ortho_increases_total():
    """total_loss must be strictly larger when lambda_ortho > 0 with non-orthogonal weights."""
    target, recon, mu, alpha, proto = _loss_inputs()
    # Create a non-trivial weight with overlapping blocks
    weight = torch.randn(INPUT_DIM, NUM_EMB * EMB_SIZE)

    out_no_ortho = compute_loss(
        target, recon, mu, alpha, proto,
        pi=0.1, gamma=0.01, beta=1.0, lambda_ent=0.01,
        lambda_ortho=0.0,
    )
    out_with_ortho = compute_loss(
        target, recon, mu, alpha, proto,
        pi=0.1, gamma=0.01, beta=1.0, lambda_ent=0.01,
        lambda_ortho=1.0,
        decoder_weight=weight,
        num_embeddings=NUM_EMB,
        embedding_size=EMB_SIZE,
    )
    assert out_with_ortho.total_loss.item() > out_no_ortho.total_loss.item()
    assert out_with_ortho.ortho_loss.item() > 0.0


def test_compute_loss_returns_all_fields():
    """LossOutput must expose all six fields."""
    target, recon, mu, alpha, proto = _loss_inputs()
    out = compute_loss(target, recon, mu, alpha, proto, pi=0.1, gamma=0.01, beta=1.0, lambda_ent=0.01)
    assert hasattr(out, "total_loss")
    assert hasattr(out, "recon_loss")
    assert hasattr(out, "cond_kl_loss")
    assert hasattr(out, "sparsity_loss")
    assert hasattr(out, "entropy_loss")
    assert hasattr(out, "ortho_loss")

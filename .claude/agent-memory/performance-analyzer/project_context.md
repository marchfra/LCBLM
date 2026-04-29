---
name: LCBLM project context
description: Key architectural facts relevant to performance analysis of LCBLM
type: project
---

Backbone: frozen Mistral-7B-v0.1, embeddings pre-extracted (D=4096).
Active experiment: `experiments/vaee_vs_sae/` comparing VAEE vs SparseAE on SST-2 token embeddings.
VAEE training loop: `exp_training.py::train_vaee()` — calls `compute_loss()` every batch, which calls `compute_decoder_ortho_loss()` when `lambda_ortho > 0`.
Typical sweep config: K=num_embeddings=128, E=embedding_size=128, D=4096.
Platform: macOS Intel (darwin 24.6.0) and Linux/ARM; torch 2.2.2 / 2.8.0 respectively.
The validation loop also calls ortho_loss (under `torch.inference_mode()`).

**Why:** The ortho loss is called every training step (train + val batches), making it a hot path.
**How to apply:** Any O(K²) operation in the loss function will dominate at K=128.

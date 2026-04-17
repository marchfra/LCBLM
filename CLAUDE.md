# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LCBLM (Learnable Concept-Base Language Model) is a Master's thesis research project. It trains Sparse AutoEncoders (SAEs) and concept-based models on language model embeddings, then evaluates them via perplexity and concept analysis. The backbone LLM (default: Mistral-7B-v0.1) is frozen; only the concept-extraction layers are trained.

**Pipeline stages** (numbered scripts):
1. Embedding extraction from backbone LLM
2. Baseline classifiers (finetuned head + linear from scratch)
3. SAE training and evaluation (TopK and Bernoulli variants)
4. LCBLM training — in progress
5. Evaluation (perplexity, concept labeling, intervenability)

## Commands

```bash
# Install dependencies
uv sync

# Lint and format
ruff check --fix --ignore=FIX002 .
ruff format .

# Type check
ty check

# Tests (suite is currently empty)
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name

# Run experiment CLIs (copy *.example.toml to *.toml first and edit as needed)
embae-exp run experiments/embae_comparison/config_mnist.toml
vaee-exp run experiments/vaee_vs_sae/config_sst2.toml

# Regenerate plots from saved results
embae-exp plot experiments/embae_comparison/experiment_outputs/.../results.json
vaee-exp plot experiments/vaee_vs_sae/experiment_outputs/.../results.json
```

Commits are validated by commitizen (conventional commit format). Pre-commit hooks run ruff and uv-lock on every commit.

After each major change to the project (new experiment, new architecture, new CLI, significant refactor), update this file to reflect the new state.

## Architecture

### Active development

Current focus is `experiments/vaee_vs_sae/`, which compares VAEE and SparseAE architectures trained on pre-extracted Mistral-7B token embeddings. Earlier work built out `src/lcblm/embedding_ae/` (`EmbeddingAE`) and `src/lcblm/vaee/` (`VAEE`), both of which are now stable. The numbered scripts in `src/lcblm/scripts/` and the notebooks have not been actively maintained and may be out of sync with the rest of the codebase.

This is a research codebase, so it intentionally contains many exploratory branches and parallel approaches.

### Core abstractions

**`src/lcblm/sae_utils/`** — Sparse AutoEncoder (foundational module)
- `model.py`: `SparseAE` — encoder → sparse activation → decoder
- `config.py`: `Config` dataclass for SAE hyperparameters
- `activations.py`: `TopK` activation
- `losses.py`: reconstruction loss, auxiliary k-sparse loss, Bernoulli KL
- `train.py`: `train_sae()` training loop
- `dataset.py`: `SAEDataset` wrapping pre-extracted embedding tensors

**`src/lcblm/embedding_ae/`** — `EmbeddingAE`, a prototype-based MLP autoencoder. Shares significant code with `sae_utils/` (pulls from it directly); a future reorganization is planned to reduce this overlap.

**`src/lcblm/vaee/`** — `VAEE`, a VAE with discrete prototype embeddings and Gumbel-Sigmoid gates. Takes a configurable `output_activation` (defaults to `nn.Identity` for unbounded inputs; pass `nn.Sigmoid()` for image data).

**`src/lcblm/utils/`** — shared utilities
- `data/`: `NextTokenDataset`, `TokenizedDataset`, `TypedDataLoader`
- `pytorch.py`: device detection, tensor helpers
- `transformers.py`: HuggingFace integration
- `typing.py` (top-level): `TensorModule`, `ShapedTensorModule`, `TypedLinear` protocols

### Experiments

Each experiment lives in its own self-contained directory with a CLI (`exp_cli.py`), config dataclasses (`exp_config.py`), and TOML config files. Use `embae_comparison` as the template for new experiment directories.

**`experiments/embae_comparison/`** (`embae-exp`) — compares `EmbeddingAE` and `SparseAE` on image datasets (MNIST, Digits). Sweeps over `n_concepts`.

**`experiments/vaee_vs_sae/`** (`vaee-exp`) — compares `VAEE` against two `SparseAE` baselines (concept-matched and parameter-matched) on pre-extracted Mistral-7B SST-2 token embeddings. Sweeps over `num_embeddings`, producing L0-MSE scatter plots and learning curves. Saves checkpoints, a fitted `StandardScaler`, and `results.json` for plot regeneration without retraining.

### Platform notes

`pyproject.toml` contains platform markers for torch: 2.2.2 on macOS Intel, 2.8.0 on Linux/ARM. These are managed with `uv` — do not edit manually without checking the markers.

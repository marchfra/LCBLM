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

# Extract LLM embeddings for a HuggingFace dataset
extract-emb --model mistralai/Mistral-7B-v0.1 --dataset SetFit/sst2 --output-dir ./embeddings/sst2_mistral
# Add --stride S to slice long documents into S-token-hop windows instead of truncating

# Train a concept model (copy *.example.toml to *.toml first and set embeddings_path)
ct-train vaee        experiments/concept_training/configs/vaee_sst2.toml
ct-train topk_sae    experiments/concept_training/configs/topk_sae_sst2.toml
ct-train sae_concept experiments/concept_training/configs/sae_concept_sst2.toml
ct-train sae_param   experiments/concept_training/configs/sae_param_sst2.toml

# Multi-run sweep: [[runs]] array in TOML trains configs sequentially
ct-train vaee        experiments/concept_training/configs/vaee_multi.toml
ct-train topk_sae    experiments/concept_training/configs/topk_sae_multi.toml

# Build concept dictionary HTML from a trained checkpoint
ct-cd --run-dir experiments/concept_training/outputs/VAEE-256x128-...

# L0-MSE scatter plot from one or more run directories
ct-plot experiments/concept_training/outputs/VAEE-50x64-... experiments/concept_training/outputs/TopK-64-SAE-4096-...
ct-plot experiments/concept_training/outputs/  # all subdirs
```

Commits are validated by commitizen (conventional commit format). Pre-commit hooks run ruff and uv-lock on every commit.

After each major change to the project (new experiment, new architecture, new CLI, significant refactor), update this file to reflect the new state.

## Architecture

### Active development

Current focus is `experiments/concept_training/`, which trains VAEE and SparseAE models on pre-extracted Mistral-7B token embeddings and supports concept dictionary analysis. Earlier work built out `src/lcblm/embedding_ae/` (`EmbeddingAE`) and `src/lcblm/vaee/` (`VAEE`), both of which are now stable. The numbered scripts in `src/lcblm/scripts/` and the notebooks have not been actively maintained and may be out of sync with the rest of the codebase.

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

**`src/lcblm/training/`** — shared training library used by `concept_training`:
- `configs.py`: `VAEEConfig`, `TopKSAEConfig`, `SAEConceptConfig`, `SAEParamConfig`, `DatasetConfig`
- `data.py`: `load_embeddings()`, `load_split()`, `save_scaler()`, `load_scaler()`
- `models.py`: `build_vaee()`, `build_sae()`, `build_ref_vaee()`, `param_matched_latent_dim()`, `resolve_latent_dim()`
- `loops.py`: `train_vaee()`, `train_topk_sae()`, `train_sae_concept()`, `train_sae_param()`, `RunResult`

**`src/lcblm/extract_embeddings.py`** — `extract-emb` CLI. Tokenises a HuggingFace dataset with a backbone LLM and saves `extracted_features_{split}.pt` files (keys: `input_ids`, `attention_masks`, `embeddings`, `word_ids`). Supports optional striding for long documents. Output is compatible with `lcblm.training.data`.

**`src/lcblm/utils/`** — shared utilities
- `data/`: `NextTokenDataset`, `TokenizedDataset`, `TypedDataLoader`
- `pytorch.py`: device detection, tensor helpers
- `transformers.py`: HuggingFace integration
- `typing.py` (top-level): `TensorModule`, `ShapedTensorModule`, `TypedLinear` protocols

### Experiments

**`experiments/concept_training/`** — canonical experiment for training and analysing concept models. Three CLIs:
- `ct-train <model> <config.toml>` — trains one model type (`vaee`, `topk_sae`, `sae_concept`, `sae_param`). Supports single-run and multi-run (`[[runs]]` array-of-tables) TOML configs. Output dirs are named `{ModelType}-{hyperparams}-{timestamp}/` and contain a checkpoint, `*_meta.json`, `scaler.pkl`, `config.json`, and `results.json`.
- `ct-cd --run-dir <outputs/...>` — loads a checkpoint and writes a self-contained HTML concept dictionary from per-token concept activations.
- `ct-plot <run_dir> [<run_dir> ...]` — reads `results.json` from each run dir and generates an L0-MSE scatter plot. Pass a parent dir to process all subdirs at once. `--csv PATH` also writes a CSV of the raw metrics.

Config files are in `experiments/concept_training/configs/` (copy `*.example.toml` → `*.toml` and set `embeddings_path`). Shared training fields (`embeddings_path`, `eos_token_id`, `n_samples`, `epochs`, `lr`, `batch_size`, `seed`, `early_stopping_*`, `wandb_project`) sit at the top level; model-specific hyperparameters go in `[[runs]]` entries (or at the top level for single-run configs).

**`experiments/embae_comparison/`** and **`experiments/vaee_vs_sae/`** and **`experiments/interpretability/`** — earlier experiments, soft-retired (directories kept for reference; entry points removed from pyproject.toml).

### Platform notes

`pyproject.toml` contains platform markers for torch: 2.2.2 on macOS Intel, 2.8.0 on Linux/ARM. These are managed with `uv` — do not edit manually without checking the markers.

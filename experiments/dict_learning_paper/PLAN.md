# VAEE as Dictionary Learning — Workshop Paper v1 Plan

## Context

The goal is to publish VAEE as a **dictionary-learning method**, *before* the larger mechanistic-interpretability paper. The framing is theoretical (probabilistic formulation of sparse dict learning with Gumbel-Sigmoid gates) plus an empirical Pareto-frontier study against standard baselines. The paper claims VAEE offers a better *joint* trade-off across reconstruction quality, per-sample sparsity (L0), total live dictionary size, and concept controllability — not necessarily a strict win on any single axis.

**Scope decisions confirmed by user:**
- Theoretical dict-learning paper, working directly on the data (with a small encoder/decoder when needed, e.g. images).
- Sampling claim is **controlled / structured sampling** only (gate concepts on/off). No FID, no full generation.
- Target: workshop paper, **1 month** to submission.
- **Team of 3**: at least one person dedicated to experiments, at least one to paper writing, one floating/lead. ~3 person-months of effort across ~4 calendar weeks.
- **No LLM / text / mechanistic-interpretability datasets in v1.** All experiments on synthetic data and images. The MI story is deferred to a v2 paper and must not bleed in.

## Honest assessment (read this before approving)

This is feasible at workshop scope but has real risks. Five I want flagged in the plan, not buried:

1. **SAE Pareto is crowded.** TopK / JumpReLU / BatchTopK / Matryoshka SAEs already cover the (L0, MSE) frontier. Expect VAEE to *match*, not crush. The paper has to be honest that the win is multi-axis (recon + L0 + total alive concepts + controllability), not single-axis.
2. **Dict-learning benchmarks aren't standardized.** No canonical setup. We pick our benchmarks; reviewers will probe the choices. Mitigation: synthetic feature-recovery benchmark with known ground truth + MIG-against-published-baselines validation step.
3. **"Controlled sampling" as a metric is soft** — addressed by metrics 6 and 7 which give it concrete quantitative form.
4. **Pixel-space dict learning is unfashionable.** Most recent literature operates on learned representations. Reviewers may ask why no natural-image experiments in pixel space. Honest answer: pixel-space MSE on natural images is a confounder (every method produces blurry results), so we chose datasets where pixel-space recon is tractable. The classical sparse-coding lineage (Olshausen, KSVD) supports this. Position the paper as "method analysis on controlled benchmarks", not "state-of-the-art on natural images".
5. **1-month timeline is aggressive.** Three people working in parallel makes it possible, but there is essentially no buffer for unexpected blockers. The plan trims sparsity sweep density (5→3 points) and cuts ablations. If a metric implementation takes more than 2 days, something has to give — likely a dataset or a metric, decided by end of week 2 if necessary.

If these five are acknowledged, the workshop paper is achievable in 4 weeks with the 3-person team.

## Datasets (4 total — fixed, no creep)

**Architectural decision (user-confirmed):** All dict-learning models operate **directly in pixel space**. No pretrained encoder, no jointly trained AE. This is the theoretically cleanest setting — dict learning on the data itself, with strong precedent in classical sparse coding (Olshausen & Field 1996, KSVD). Datasets are deliberately chosen to keep pixel-space reconstruction tractable: low/medium dimensionality and either simple statistics (MNIST/Fashion-MNIST) or binary content (dSprites).

| Dataset | Role | Input | Dim | Why tractable in pixel space |
|---|---|---|---|---|
| **Synthetic superposition** (Anthropic-style: known sparse features in high-d Gaussian) | Ground-truth feature recovery | Raw vector | 256 (configurable) | Vector data by construction |
| **MNIST** | Cheap sanity check; concepts have clear visual identity (strokes, digits) | Raw pixels, flattened | 784 | Low-dim grayscale, near-binary |
| **Fashion-MNIST** | Step up from MNIST; clothing categories have richer structure but stay in the same dim | Raw pixels, flattened | 784 | Same dim/format as MNIST, harder content |
| **dSprites** | Disentanglement benchmark with **5 ground-truth factors** (shape, scale, rotation, x, y) | Raw pixels, flattened | 4096 (64×64 binary) | Binary geometric shapes — pixel-space MSE works cleanly despite higher dim |

**Why this set (replaces earlier CIFAR + CelebA plan):**
- dSprites replaces CelebA: same controllability/disentanglement role, but with **clean ground-truth factors** rather than noisy 40-attribute labels, and tractable pixel-space recon (binary). The standard disentanglement metrics (MIG, DCI, FactorVAE-score, SAP) are well-defined here.
- Fashion-MNIST replaces CIFAR: provides "more semantic complexity than MNIST" without natural-image-statistics pixel-recon difficulty. If reviewers ask for natural-image evidence, we can mention CIFAR as future work or add it in week-12 buffer.
- Deliberately avoided: any dataset where pixel-space MSE is fundamentally noisy (CIFAR, CelebA at any resolution). The paper sidesteps this confounder.

**Alternative if user prefers only 3 datasets:** Drop Fashion-MNIST → synthetic + MNIST + dSprites is a defensible workshop scope (ground-truth synthetic, visual sanity check, disentanglement headline). Slightly thinner empirical evidence; significantly less to write up. Marked as a fallback if Fashion-MNIST adds little signal in week 3.

**Compute envelope** (revised for 1-month timeline):
- 5 models × 3 sparsity points × 4 datasets = 60 runs.
- MNIST + Fashion-MNIST + synthetic runs are minutes each → <1 GPU-day for 45 of the 60 runs.
- dSprites runs are ~1-2 hours each → ~1-2 GPU-days for the 15 dSprites runs.
- ResNet feature extraction for intervention metric: <1 hour, done once on validation sets only.
- dSprites factor regressor: ~30 minutes, trained once.
- **Total: ~2–3 GPU-days.** Fits inside week 2 even on a single GPU.

**Optional 5th dataset (only if multiple weeks of buffer remain):** patch-based natural image sparse coding (Olshausen 16×16 patches from BSDS or similar). Connects the work to the classical sparse-coding literature. Mentioned only as buffer.

## Baselines (4 total — fixed)

| Baseline | Why it's in |
|---|---|
| **L1-SAE** | Standard sparse dict learning baseline |
| **TopK-SAE** | Strongest modern SAE baseline; already implemented in `src/lcblm/sae_utils/` |
| **VQ-VAE** | The natural "discrete-codebook" comparison; tests whether soft Gumbel-Sigmoid beats hard quantization. **Wrapped without its native conv enc/dec** — VQ codebook applied directly to raw pixels (consistent with VAEE/SAE setup). Note this in the paper as the head-to-head-fair variant. |
| **β-VAE** | The "continuous-latent VAE" comparison; tests whether discreteness via gates matters at all |

**Deferred** (mention in related work, don't implement): JumpReLU, BatchTopK, Matryoshka SAE, FactorVAE, Gated SAE. If 2 weeks of buffer appear, add JumpReLU-SAE; otherwise leave them.

## Metrics

**Seven headline metrics, mapped to the claims.** Everything else is a sanity check or qualitative figure.

| # | Metric | Claim defended | Datasets | Implementation |
|---|---|---|---|---|
| 1 | **Reconstruction MSE** (also reported as R² for intuition) | Better reconstruction | All 4 | Already in `RunResult.best_val_recon`; add R² as a derived field |
| 2 | **Per-sample L0** | Fewer active concepts per sample | All 4 | Already in `RunResult.best_l0` |
| 3 | **Total alive dictionary size** (concepts firing on ≥1% of validation samples) | Fewer total latent dims — **strongest claim** | All 4 | New field in `RunResult`; computed at end-of-training pass |
| 4 | **Class purity** (per-concept label entropy of top-K activating samples) | Quantitative concept interpretability | MNIST, Fashion-MNIST | New: `eval/metrics.py:class_purity` |
| 5 | **MIG** (Mutual Information Gap vs the 5 ground-truth factors) — *correlational* | Disentanglement (activations correlate with factors) | dSprites | New: `eval/disentanglement.py:mig` — single metric, no DCI/FactorVAE/SAP. Standard direction: for each factor, find the latent with highest MI (gap = top1 − top2, normalised by H(factor)). **Supplementary:** also compute inverted MIG (for each alive latent, find the factor it captures — gap normalised by H(factor_top1)); if VAEE's alive latents each map cleanly to one factor, this tells the "efficient dictionary" story. Report inverted MIG only if the numbers support it. |
| 6 | **Intervention consistency in ResNet feature space** (mean cosine similarity of gate-flip directions across N=256 validation inputs, in frozen ResNet-50 features) | Controlled sampling / interpretable intervention | All 3 image datasets | New: `eval/intervention.py:consistency_resnet` |
| 7 | **Intervention-based factor recovery (dSprites)** — *causal* counterpart to MIG. For each concept g and each factor f ∈ {shape, scale, rotation, x, y}, intervene on g across N inputs, decode, and measure the induced change in f (via direct pixel comparison or a small pretrained factor regressor on dSprites). Report: per-concept dominant factor + dominance ratio (how much the strongest factor changes vs others). Companion *qualitative* artifact: a grid showing intervention before/after with the corresponding factor annotated. **Doubles as an in-distribution-preservation check**: if interventions take the recon OOD, the factor regressor reads noise and dominance ratios collapse — high scores on this metric inherently certify that the model's interventions stay on the data manifold. | Causal feature recovery + in-distribution preservation (a single metric covers both) | dSprites | New: `eval/intervention.py:factor_recovery_dsprites` |

**Headline figure:** L0-MSE Pareto curves per dataset (5 model traces). Pareto sweeps come from varying the sparsity hyperparameter per model (π for VAEE, k for TopK-SAE, λ for L1-SAE, β for β-VAE, codebook size for VQ-VAE). **3 sparsity points per model per dataset** (compressed from 5 for the 1-month timeline; can expand to 5 in a v2 / camera-ready if time allows).

**Sanity checks (single-line claims, not headline metrics):**
- **Feature recovery on synthetic** — verify VAEE and VQ-VAE recover ground-truth features at >0.9 cosine similarity (Hungarian matching). One line in results: "Method works as intended on a controlled benchmark." Detail goes in appendix.
- **β-VAE MIG baseline check** — our MIG implementation should reproduce published β-VAE numbers on dSprites (≈0.2-0.3 at β=4). Validates the metric, not a comparison.

**Qualitative artifacts** (all datasets):
- Concept dictionary HTML via `ct-cd` (top-K activating samples per concept)
- Latent traversal grids (pairs visually with the intervention-consistency metric — same concept flips, before/after image)

**ResNet usage caveat** (must be stated clearly in the paper): The frozen ResNet-50 is used only in metric #6 (intervention consistency in semantic space) as an evaluation yardstick. It is *not* part of any trained model, never sees the target datasets at training time, and is identical across all 5 compared models — so it cannot bias the comparison. Dict learning itself happens entirely in pixel space.

**Cut from earlier drafts** (mention in "future work" or appendix only if asked): dead-feature rate, DCI / FactorVAE / SAP, concept-activation entropy, concept reuse Gini, pixel-space intervention consistency, seed stability. In-distribution preservation is no longer separately tracked because metric #7 (intervention-based factor recovery on dSprites) implicitly subsumes it.

## Code structure

Keep current layout, add three things:

```
src/lcblm/
  baselines/                       # NEW
    vq_vae.py                      # VQ codebook layer (no internal enc/dec — applied directly to pixels)
    beta_vae.py                    # β-VAE latent layer (continuous z + KL, no internal enc/dec)
    __init__.py
  data/                            # NEW
    synthetic.py                   # configurable sparse-feature generator
    image_loaders.py               # MNIST / Fashion-MNIST / dSprites — flatten to raw-pixel tensors
  eval/                            # NEW — only what's needed for the 7 headline metrics + 2 sanity checks
    metrics.py                     # alive_dict_size, class_purity, feature_recovery (sanity)
    disentanglement.py             # mig(latent_activations, factors) → float; also inverted_mig
    intervention.py                # consistency_resnet (metric 6), factor_recovery_dsprites (metric 7)
    resnet_eval.py                 # frozen ResNet-50 feature extractor used *only* by intervention.py
    dsprites_factors.py            # small factor regressor for dSprites, trained once from ground-truth labels
  training/
    configs.py                     # add VQVAEConfig, BetaVAEConfig, ImageDatasetConfig
    loops.py                       # add train_vq_vae, train_beta_vae; extend RunResult
    models.py                      # add build_vq_vae, build_beta_vae
experiments/
  dict_learning_paper/             # NEW — paper-only configs and runners
    configs/
      synthetic_*.toml
      mnist_*.toml
      fmnist_*.toml
      dsprites_*.toml
    run_all.sh                     # reproducibility entry point
    figures/                       # generated plots

# ct-eval: single post-training evaluation script
# Usage: ct-eval --run-dir <outputs/...> --dataset <synthetic|mnist|fmnist|dsprites>
#                [--data-path <path>]   # for dSprites npz or image root
#                [--labels]             # pass for MNIST/FashionMNIST (enables class purity)
#                [--device cpu]
#
# Outputs a single eval_results.json in the run dir with ALL metrics:
#
#   From RunResult (already collected during training):
#     best_val_recon, best_l0, alive_dict_size
#
#   Computed post-training over the val set:
#     class_purity          (MNIST, FashionMNIST only — requires labels)
#     feature_recovery      (synthetic only — requires ground-truth features matrix)
#     mig                   (dSprites only)
#     inverted_mig          (dSprites only — supplementary)
#     intervention_consistency_resnet   (all image datasets)
#     factor_recovery       (dSprites only)
#
# All post-training metrics load the best checkpoint, run the val set through the
# model once (no gradients), and use the resulting latent activations. The script
# detects which metrics are applicable from the dataset name and available files.
```

**Reused existing code (do not duplicate):**
- `src/lcblm/training/loops.py:train_vaee` and friends — extend pattern, do not rewrite
- `src/lcblm/training/loops.py:RunResult` — add fields rather than introducing a new schema
- `ct-plot` (`src/lcblm/scripts/plot.py`) — extend with VQ-VAE/β-VAE colors; the L0-MSE scatter is the paper's headline figure
- `ct-cd` (`src/lcblm/scripts/build_cd.py`) — extend `ModelAdapter` for VQ-VAE/β-VAE; add an image-domain rendering mode (top-K activating images per concept, instead of top-K tokens)
- `ct-eval` (`experiments/dict_learning_paper/eval.py`) — **single post-training eval script**; loads a run dir, reports all metrics (training + post-training) into `eval_results.json`. Detects applicable metrics from dataset name.
- `src/lcblm/vaee/models.py` — VAEE core untouched; input is just flattened pixels for image datasets
- **`extract-emb` is unused in v1** — no text data this round
- **No encoder/decoder modules** — VAEE/SAE/VQ-VAE/β-VAE all consume flattened pixels directly

## Repository branch strategy

Single long-lived paper branch off `main`, with topic branches merging into it. Avoid feature branches off `main` for paper-specific work — they pollute the research branch.

```
main                          (research codebase, ongoing)
└── paper/dict-learning-v1    (paper integration branch)
    ├── paper/baselines       (VQ-VAE, β-VAE implementations)
    ├── paper/eval            (eval harness + metrics)
    ├── paper/synthetic       (synthetic dataset + ground-truth recovery)
    └── paper/figures         (figure generation only — no model code)
```

Merge topic branches into `paper/dict-learning-v1` as they stabilize. Once the paper is submitted, squash-merge `paper/dict-learning-v1` back to `main`.

## Timeline (4 weeks, 3 parallel tracks)

Three tracks run in parallel; sync at end of each week. Track A is the experiment owner, Track B is the writing owner, Track C is the floating lead/infra/QA.

### Track A — Experiments (1 person, full-time)

**Week 1 — Baselines, data, core metrics**
- Days 1-2: Implement `VQVAE` + `BetaVAE` as dict-learning layers + their `train_*` loops + configs (using existing VAEE/SAE training patterns as templates).
- Days 3-4: Implement `src/lcblm/data/synthetic.py`, `src/lcblm/data/image_loaders.py`. Smoke-test all 4 datasets feed into all 5 models without error.
- Days 4-5: Implement metrics 1-4 + feature-recovery sanity check in `src/lcblm/eval/metrics.py`. Extend `RunResult`.
- Days 6-7: Implement Pareto-sweep driver. Kick off synthetic + MNIST sweeps.

**Week 2 — All sweeps + intervention infra**
- Days 1-3: Run all 4 dataset Pareto sweeps (60 runs, ~2 GPU-days). Background while implementing remaining metrics.
- Days 3-5: Implement metric 5 (MIG) + metric 6 (intervention consistency in ResNet space) + metric 7 (dSprites factor recovery). Train the dSprites factor regressor.
- Days 5-7: Compute all 7 metrics for all completed runs. Generate first-pass Pareto figures.

**Week 3 — Figures + iteration**
- Days 1-3: Generate per-dataset concept-dictionary HTMLs. Select figures for the paper.
- Days 3-5: Re-run anything that looks broken; backfill missing metric computations.
- Days 6-7: Hand off final tables + figures to Track B.

**Week 4 — Buffer + reviewer-anticipated ablations**
- Days 1-3: Camera-ready quality figures. One ablation (e.g., VAEE π-sweep on dSprites at finer granularity) if useful.
- Days 4-7: Support Track B with figure tweaks, supplementary results, revisions.

### Track B — Writing (1 person, full-time)

**Week 1**
- Day 1: **Produce a detailed paper-structure plan before any drafting begins.** Section-by-section outline of the workshop paper: what each section claims/argues, how existing material from `probabilistic_formulation_v7.pdf` + `research_note.pdf` maps into it, what's missing and needs to be written from scratch, and where the 7 headline metrics and 4 datasets land. User approves the plan before tex work starts. No prose, no figure placeholders, no tex edits until this is approved.
- Days 2-3: Set up paper template, figure placeholders. Identify target workshop and confirm format.
- Days 3-7: Draft theory section based on `probabilistic_formulation_v7.pdf` + `research_note.pdf`. Outline empirical section.

**Week 2**
- Days 1-4: Related work + introduction + abstract draft.
- Days 5-7: Empirical section scaffolding with placeholders for the 7 metrics × 4 datasets.

**Week 3**
- Days 1-4: Fill in empirical section as results arrive from Track A.
- Days 5-7: Limitations + future-work + final discussion.

**Week 4**
- Days 1-5: Full revision passes, polish, double-check claims against the data.
- Days 6-7: Submit.

### Track C — Floating lead / infra / QA (1 person, full-time or part-time)

**Week 1**: Pair-program with Track A on whichever component is critical-path (probably the baseline implementations or the metric implementations). Code review.

**Week 2**: Run validation checks on metrics — verify MIG implementation reproduces published β-VAE numbers on dSprites; verify intervention-consistency on a hand-constructed test case; verify feature recovery on synthetic. Catch metric bugs before they pollute results.

**Week 3**: Pair with Track B on the empirical section — interpret results, flag any numbers that look too good or too bad, suggest framings. Verify final figures match the underlying CSVs.

**Week 4**: Final review of the submitted paper. Manage submission logistics.

### What gets cut to fit 1 month

- Pareto sweep density: **5 → 3 sparsity points** per model. Most consequential cut; trims compute and run count by 40%.
- Seed-stability runs (already cut from the 7 headline metrics).
- Multi-baseline ablations (JumpReLU SAE etc.) — workshop scope, mentioned in related work only.
- Latent-traversal figures restricted to **dSprites and one other dataset** (probably Fashion-MNIST) rather than all 4.
- W&B per-step logging kept but no per-run dashboards / writeups.

### Hard constraints that *cannot* slip

- All 7 metrics must be computed on all relevant datasets by end of week 2.
- Track B must have at least skeleton + theory draft by end of week 2 (don't wait for results).
- Submission target: end of week 4. No "I'll need 1 more day" — build in week 4 as the buffer.

## Realistic expected results

For each of the 6 headline metrics:

| Metric | Expected outcome | Confidence |
|---|---|---|
| **Reconstruction MSE** at given L0 (MNIST, Fashion-MNIST) | VAEE within ~5% of TopK-SAE on the Pareto curve. Not a clean win. | High |
| **Reconstruction MSE** at given L0 (dSprites) | All five models reach near-perfect recon (binary content); curve mostly flat. The interesting axis on dSprites is L0 + MIG, not MSE. | High |
| **Per-sample L0** | Standard plotting axis; not a "winner" metric on its own. The headline is the joint (L0, MSE) curve. | n/a |
| **Total alive dictionary size** | **VAEE clear win.** Gate-KL actively shrinks the dictionary; SAE L1/TopK do not. Probably the strongest result. | High |
| **Class purity** (MNIST, Fashion-MNIST) | Likely VAEE > TopK-SAE > L1-SAE on mean purity; β-VAE highest entanglement (lowest purity); VQ-VAE strong on purity but concentrated in fewer codes. | Moderate |
| **MIG** (dSprites) | β-VAE is the canonical baseline; published scores ≈0.2-0.3 at β=4. VAEE plausibly matches β-VAE on MIG while using fewer active concepts. Worst case: VAEE loses on MIG but wins on alive-dict-size — still a defensible multi-axis story. | Moderate-low |
| **Intervention consistency in ResNet space** | **VAEE clear win.** Single-gate flip is well-defined; for SAE you'd zero a latent (less semantically clean), for VQ-VAE you'd swap a code (discrete, brittle), for β-VAE concepts are entangled. | High |
| **Intervention-based factor recovery (dSprites)** | The interesting comparison vs MIG: a model can have high MIG (correlation) but low causal control. **VAEE plausibly leads on causal control** because its gates literally turn concepts on/off — interventions are by construction discrete. β-VAE may have similar MIG but weaker causal control because its latents are entangled in the decoder. SAE/TopK do *not* have a natural intervention semantic. This metric is where VAEE's architectural choice (gates) pays off most clearly. | Moderate-high |

**Headline framing for the paper:** "VAEE achieves the **smallest live dictionary** for a given (per-sample sparsity, reconstruction) operating point, supports **interpretable single-concept intervention** out of the box, and is competitive with β-VAE on disentanglement (MIG) while using substantially fewer active concepts. It does not strictly dominate SAEs on the (L0, MSE) frontier alone, but extends the relevant trade-off space to include dictionary size, controllability, and disentanglement."

## Verification / how to test end-to-end

1. **Unit tests** for new metrics (`tests/eval/test_metrics.py`, `tests/eval/test_intervention.py`, `tests/eval/test_disentanglement.py`) — feature_recovery on a synthetic example with known matching; intervention_consistency on a hand-constructed model with known direction; disentanglement metrics against published reference values for β-VAE on dSprites.
2. **Smoke runs** — each new baseline trains for 5 epochs on MNIST without error; `RunResult` populates all new fields.
3. **Reproducibility** — `experiments/dict_learning_paper/run_all.sh` reproduces every figure from a clean checkout. Pin a `requirements.lock` or rely on existing `uv.lock`.
4. **L0-MSE Pareto figure** — `ct-plot experiments/dict_learning_paper/outputs/` regenerates the headline figure (Pareto curves across all 5 models × 4 datasets).
5. **Sanity check on synthetic** — ground-truth features should be recovered at >0.9 cosine similarity by at least VAEE and VQ-VAE; failure on this benchmark means the formulation is wrong, not that the experiment failed.
6. **Disentanglement sanity** — β-VAE on dSprites should achieve published MIG range (≈0.2-0.3 at β=4); deviation indicates a metric-implementation bug.

## Out of scope for v1 (explicitly)

- **All LLM / text data and embeddings** — no SST-2, no token-level activations, no LM perplexity. Avoids any MI flavor in v1.
- LLM-perplexity-with-SAE-in-loop evaluation (defer to MI paper).
- Steering / activation patching experiments (defer to MI paper).
- Full FID-style unconditional generation (user-confirmed out of scope).
- JumpReLU / Matryoshka / BatchTopK / Gated SAE baselines (mentioned in related work only).
- Audio, video, multimodal data.

## Critical files (to edit)

- `src/lcblm/training/configs.py` — add `VQVAEConfig`, `BetaVAEConfig`, `ImageDatasetConfig`
- `src/lcblm/training/models.py` — add `build_vq_vae`, `build_beta_vae`; wire image enc/dec around all 4 model types
- `src/lcblm/training/loops.py` — add `train_vq_vae`, `train_beta_vae`; extend `RunResult` with `alive_dict_size` and `r2` (the new headline fields)
- `src/lcblm/baselines/vq_vae.py`, `src/lcblm/baselines/beta_vae.py` — NEW (dict-learning layers, no internal enc/dec)
- `src/lcblm/data/synthetic.py`, `src/lcblm/data/image_loaders.py` — NEW (MNIST / Fashion-MNIST / dSprites + synthetic generator)
- `src/lcblm/eval/metrics.py` — NEW (alive_concepts, feature_recovery, reuse_gini, r2, class_purity, activation_entropy)
- `src/lcblm/eval/disentanglement.py` — NEW (MIG, DCI, FactorVAE-score, SAP — dSprites only)
- `src/lcblm/eval/intervention.py` — NEW (metric 6: gate-flip consistency in ResNet feature space; metric 7: dSprites factor recovery + in-distribution preservation as a side effect)
- `src/lcblm/eval/resnet_eval.py` — NEW (frozen ResNet-50 used only by `intervention.py`, never inside a trained model)
- `src/lcblm/eval/dsprites_factors.py` — NEW (small factor regressor trained from dSprites ground-truth labels; used only by metric 7)
- `src/lcblm/eval/synthetic_runner.py`, `src/lcblm/eval/image_runner.py`, `src/lcblm/eval/nn_lookup.py` — NEW
- `src/lcblm/scripts/plot.py` — extend with new model colors
- `src/lcblm/scripts/build_cd.py` — add image-domain rendering mode (top-K activating images per concept)
- `experiments/dict_learning_paper/` — NEW directory with configs and `run_all.sh`
- `tests/eval/test_metrics.py` — NEW unit tests
- Documentation: extend `CLAUDE.md` to mention the paper experiment layout (per the repo convention "after each major change ... update this file").

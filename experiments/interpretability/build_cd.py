"""Concept dictionary builder CLI.

Loads a trained model checkpoint from a run directory produced by train.py,
extracts per-token concept activations over a dataset split, and writes a
self-contained HTML concept dictionary.

The model type and all hyperparameters are read from the *_meta.json file
saved alongside each checkpoint — no flags required beyond the run directory
and model name.

Usage
-----
    interp-cd --run-dir outputs/VAEE-256x128_20260430_120000
    interp-cd --run-dir outputs/TopK-SAE-16384_20260430_120000 --split train
    interp-cd --run-dir outputs/SAE-256_20260430_120000 --top-k 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.interpretability.data import load_scaler, load_split
from experiments.interpretability.model_adapter import (
    ModelAdapter,
    SparseAEAdapter,
    VAEEAdapter,
    get_token_activations,
)
from lcblm.analysis import aggregate_to_words, build_concept_dictionary
from lcblm.sae_utils import SparseAE, TopK
from lcblm.vaee.models import VAEE

# ── Model loading ─────────────────────────────────────────────────────────────


def _load_vaee(state_dict: dict, meta: dict) -> VAEE:
    encoder_type = meta.get("encoder_type", "shallow")
    model = VAEE(
        input_dim=meta["input_dim"],
        hidden_dim=meta.get("hidden_dim", 256),
        num_embeddings=meta["num_embeddings"],
        embedding_size=meta["embedding_size"],
        output_activation=None,
        encoder_type=encoder_type,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_sparse_ae(state_dict: dict, meta: dict) -> SparseAE:
    if meta["model_type"] == "topk_sae":
        activation: nn.Module = TopK(meta["topk_k"])
    else:
        activation = nn.ReLU()

    model = SparseAE(
        input_dim=meta["input_dim"],
        latent_dim=meta["latent_dim"],
        activation=activation,
    )
    # tied_bias is a Parameter registered only after init_tied_bias() is called.
    # Create a placeholder so load_state_dict can overwrite it.
    model.init_tied_bias(torch.zeros(meta["input_dim"]))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_adapter(ckpt_path: Path) -> ModelAdapter:
    meta_path = ckpt_path.with_name(ckpt_path.stem + "_meta.json")
    if not meta_path.exists():
        msg = f"Metadata file not found: {meta_path}"
        raise FileNotFoundError(msg)

    with meta_path.open() as f:
        meta = json.load(f)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_type = meta["model_type"]

    if model_type == "vaee":
        return VAEEAdapter(_load_vaee(state_dict, meta))
    if model_type in ("topk_sae", "sae_concept", "sae_param"):
        return SparseAEAdapter(_load_sparse_ae(state_dict, meta))

    msg = f"Unknown model_type in metadata: {model_type!r}"
    raise ValueError(msg)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="interp-cd",
        description="Build an HTML concept dictionary from a trained checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        "-r",
        required=True,
        help="Run output directory produced by interp-train.",
    )
    parser.add_argument(
        "--split",
        "-s",
        default="val",
        choices=["train", "val"],
        help="Data split to compute activations on (default: val).",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=15,
        help="Token types / sentences to show per concept (default: 15).",
    )
    parser.add_argument(
        "--max-concepts",
        "-n",
        type=int,
        default=None,
        help="Maximum number of concepts to display (default: all).",
    )
    parser.add_argument(
        "--context-size",
        "-c",
        type=int,
        default=4,
        help="Tokens to show on each side of the target in the token view (default: 4).",  # noqa: E501
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Activation threshold for L0 computation. Defaults to the model's "
        "built-in default (0.5 for VAEE, 0.0 for SAE).",
    )
    parser.add_argument(
        "--tokenizer",
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace tokenizer name or path (default: mistralai/Mistral-7B-v0.1).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output HTML path. Defaults to {run_dir}/{model}_{split}_cd.html.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_candidates = [p for p in ckpt_dir.glob("*.pt") if not p.stem.endswith("_meta")]
    if len(ckpt_candidates) == 0:
        msg = f"No checkpoint found in {ckpt_dir}"
        raise FileNotFoundError(msg)
    if len(ckpt_candidates) > 1:
        ckpt_path = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
    else:
        ckpt_path = ckpt_candidates[0]

    config_path = run_dir / "config.json"
    if not config_path.exists():
        msg = f"config.json not found in {run_dir}"
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        config = json.load(f)
    ds_cfg = config["dataset_config"]

    print(f"Loading checkpoint: {ckpt_path.name}")
    adapter = load_adapter(ckpt_path)
    threshold = (
        int(args.threshold) if args.threshold is not None else adapter.default_threshold
    )
    print(
        f"  model: {ckpt_path.stem}"
        f"  n_concepts={adapter.n_concepts}"
        f"  threshold={threshold}",
    )

    print(f"Loading {args.split} embeddings...")
    scaler = load_scaler(run_dir / "scaler.pkl")
    dataset = load_split(
        embeddings_path=ds_cfg["embeddings_path"],
        split=args.split,
        scaler=scaler,
        eos_token_id=ds_cfg["eos_token_id"],
        n_samples=ds_cfg.get("n_samples", -1),
    )
    print(f"  {dataset.num_sentences} sentences")

    print("Running inference...")
    alpha, token_ids, sentence_indices, positions = get_token_activations(
        adapter,
        dataset,
    )
    mean_l0 = (alpha > threshold).sum(axis=1).mean()
    print(f"  {len(token_ids)} tokens  mean L0={mean_l0:.2f}")

    word_alpha = word_token_ids_arr = word_sentence_indices = word_positions = None
    if dataset.word_ids is not None:
        flat_word_ids = dataset.word_ids[
            torch.tensor(sentence_indices, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
        ].numpy()
        word_alpha, word_token_ids_arr, word_sentence_indices, word_positions = (
            aggregate_to_words(
                alpha,
                token_ids,
                sentence_indices,
                positions,
                flat_word_ids,
            )
        )
        print(f"  {len(word_token_ids_arr)} words")

    print(f"Loading tokenizer {args.tokenizer}...")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.tokenizer)  # ty:ignore[invalid-assignment]

    out_path = (
        Path(args.out)
        if args.out
        else run_dir / f"{ckpt_path.stem}_{args.split}_cd.html"
    )
    title = f"{ckpt_path.stem} Concept Dictionary — {ds_cfg['name']} {args.split}"

    print("Building concept dictionary...")
    build_concept_dictionary(
        alpha=alpha,
        token_ids=token_ids,
        sentence_indices=sentence_indices,
        positions=positions,
        dataset=dataset,
        tokenizer=tokenizer,
        num_concepts=adapter.n_concepts,
        threshold=threshold,
        top_k=args.top_k,
        max_concepts=args.max_concepts,
        context_size=args.context_size,
        title=title,
        out_path=out_path,
        word_alpha=word_alpha,
        word_token_ids=word_token_ids_arr,
        word_sentence_indices=word_sentence_indices,
        word_positions=word_positions,
    )


if __name__ == "__main__":
    main()

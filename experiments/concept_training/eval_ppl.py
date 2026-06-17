"""Perplexity evaluation CLI for concept model configurations.

Generates text autoregressively using Mistral-7B with one of:
  - the original lm_head (baseline)
  - a trained SAE/VAEE reconstruction bottleneck + original lm_head (per run-dir)

Generated texts are scored with a separate perplexity LLM.

Usage
-----
    ct-eval-ppl --out-dir results/ppl
    ct-eval-ppl -r outputs/TopK-64-SAE-16384-... --num-samples 50 --seed 7
    ct-eval-ppl -r outputs/TopK-64-... -r outputs/VAEE-256x128-... --out-dir results/ppl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import evaluate
import torch
from torch import Tensor, nn
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.concept_training.build_cd import load_adapter
from lcblm.training.data import load_scaler
from lcblm.utils.seed import set_seeds

if TYPE_CHECKING:
    from sklearn.preprocessing import StandardScaler

    from experiments.concept_training.model_adapter import ModelAdapter

BACKBONE_ID = "mistralai/Mistral-7B-v0.1"


# ── Wrappers ──────────────────────────────────────────────────────────────────


class ScaledReconHead(nn.Module):
    """Backbone embeddings → scaler → adapter encode+decode → inv-scaler → lm_head."""

    def __init__(
        self,
        adapter: ModelAdapter,
        lm_head: nn.Module,
        scaler: StandardScaler,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.lm_head = lm_head
        self.scaler = scaler
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d).float().cpu().numpy()
        norm = torch.from_numpy(self.scaler.transform(flat).astype("float32")).to(
            self.device,
        )
        with torch.inference_mode():
            recon = self.adapter.decode(self.adapter.encode(norm))
        inv = torch.from_numpy(
            self.scaler.inverse_transform(recon.float().cpu().numpy()).astype(
                "float32",
            ),
        ).to(self.device)
        return self.lm_head(inv.reshape(b, t, d))


# ── Generation ────────────────────────────────────────────────────────────────


def _top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
    filter_value: float = float("-inf"),
) -> Tensor:
    if top_k > 0:
        threshold = torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
        logits[logits < threshold] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1,
        )
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0][indices_to_remove] = filter_value

    return logits


def _generate_one(  # noqa: PLR0913
    input_ids: Tensor,
    backbone: nn.Module,
    head: nn.Module,
    eos_token_id: int,
    max_length: int,
    temp: float,
    topk: int,
    topp: float,
    rep_penalty: float,
) -> Tensor:
    past_key_values = None
    for _ in range(max_length):
        lm_out = backbone(
            input_ids[:, -1:] if past_key_values is not None else input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = lm_out.past_key_values
        features = lm_out.last_hidden_state.float()
        logits = head(features)

        # repetition penalty
        score = logits[:, -1, input_ids[0]]
        score = torch.where(score < 0, score * rep_penalty, score / rep_penalty)
        logits[:, -1, input_ids[0]] = score

        next_token_logits = logits[:, -1, :] / temp
        filtered = _top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
        next_token = torch.multinomial(
            torch.softmax(filtered, dim=-1),
            num_samples=1,
        )
        input_ids = torch.cat((input_ids, next_token), dim=-1)
        if next_token.item() == eos_token_id:
            break

    return input_ids


def generate_texts(  # noqa: PLR0913
    backbone: nn.Module,
    head: nn.Module,
    tokenizer: object,
    num_samples: int,
    seed: int,
    max_length: int = 100,
    temp: float = 0.7,
    topk: int = 100,
    topp: float = 0.9,
    rep_penalty: float = 1.5,
    eos_token_id: int | None = None,
) -> list[str]:
    set_seeds(seed)
    backbone.eval()
    head.eval()

    device = next(
        (p.device for p in head.parameters() if p.device.type != "cpu"),
        next(head.parameters()).device,
    )
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id  # type: ignore[union-attr]

    start_ids = tokenizer.encode("", return_tensors="pt").to(device)  # type: ignore[union-attr]

    texts: list[str] = []
    with torch.inference_mode():
        for _ in trange(num_samples, desc="Generating", unit="sample"):
            out_ids = _generate_one(
                input_ids=start_ids,
                backbone=backbone,
                head=head,
                eos_token_id=eos_token_id,
                max_length=max_length,
                temp=temp,
                topk=topk,
                topp=topp,
                rep_penalty=rep_penalty,
            )
            texts.append(tokenizer.decode(out_ids[0]))  # type: ignore[union-attr]

    return texts


# ── Perplexity ────────────────────────────────────────────────────────────────


def _compute_perplexity(
    texts: list[str],
    ppl_model: nn.Module,
    ppl_tokenizer: object,
    max_length: int,
) -> float:
    metric = evaluate.load(
        str(Path(__file__).parent / "perplexity.py"),
        module_type="metric",
    )
    metric.add_batch(predictions=texts)
    result = metric.compute(
        model=ppl_model,
        tokenizer=ppl_tokenizer,
        max_length=max_length,
    )
    return float(result["mean_perplexity"])  # type: ignore[index]


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate texts and evaluate perplexity for concept model configs.",
    )
    p.add_argument(
        "--run-dir",
        "-r",
        nargs="+",
        type=Path,
        default=None,
        metavar="DIR",
        help="Run dir(s) from ct-train. Each adds one bottleneck config.",
    )
    p.add_argument(
        "--ppl-model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID used to score perplexity.",
    )
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=100, help="Max tokens per sample.")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--topk-sampling", type=int, default=100, dest="topk")
    p.add_argument("--topp", type=float, default=0.9)
    p.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("ppl_results"),
    )
    return p.parse_args()


def _generate_and_save(  # noqa: PLR0913
    label: str,
    backbone: nn.Module,
    head: nn.Module,
    tokenizer: object,
    out_dir: Path,
    num_samples: int,
    seed: int,
    gen_kwargs: dict,
) -> Path:
    print(f"\n[{label}]")
    texts = generate_texts(
        backbone=backbone,
        head=head,
        tokenizer=tokenizer,
        num_samples=num_samples,
        seed=seed,
        **gen_kwargs,
    )
    out_file = out_dir / f"generated_texts_{label}.json"
    with out_file.open("w") as f:
        json.dump(texts, f, indent=2)
    print(f"  Saved {len(texts)} texts → {out_file}")
    return out_file


def _load_backbone(device: torch.device) -> tuple[nn.Module, nn.Module, object, int]:
    print(f"Loading {BACKBONE_ID} …")
    full_lm = AutoModelForCausalLM.from_pretrained(
        BACKBONE_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    full_lm.eval()
    backbone = full_lm.model
    lm_head = full_lm.lm_head.to(device, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_ID)
    tokenizer.pad_token = tokenizer.eos_token
    eos_token_id: int = tokenizer.eos_token_id  # type: ignore[assignment]
    return full_lm, backbone, lm_head, tokenizer, eos_token_id  # type: ignore[return-value]


def _score_all(
    configs: list[tuple[str, Path]],
    ppl_model_id: str,
    max_length: int,
    out_dir: Path,
) -> dict[str, float]:
    print(f"\nLoading perplexity model {ppl_model_id} …")
    ppl_lm = AutoModelForCausalLM.from_pretrained(
        ppl_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    ppl_lm.eval()
    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_id)
    ppl_tokenizer.pad_token = ppl_tokenizer.eos_token

    ppls: dict[str, float] = {}
    for config_name, texts_file in configs:
        print(f"\n[{config_name}] computing perplexity …")
        with texts_file.open() as f:
            texts: list[str] = json.load(f)
        ppls[config_name] = _compute_perplexity(
            texts=texts,
            ppl_model=ppl_lm,
            ppl_tokenizer=ppl_tokenizer,
            max_length=max_length,
        )
        print(f"  mean perplexity = {ppls[config_name]:.2f}")

    ppl_file = out_dir / "perplexities.json"
    with ppl_file.open("w") as f:
        json.dump(ppls, f, indent=2)
    print(f"\nPerplexities saved → {ppl_file}")
    return ppls


def _print_table(ppls: dict[str, float]) -> None:
    col1 = max(len(k) for k in ppls)
    col2 = len("Perplexity")
    sep = "=" * (col1 + col2 + 3)
    print(f"\n{sep}")
    print(f"{'Config':<{col1}} | {'Perplexity':>{col2}}")
    print(sep)
    for name, val in sorted(ppls.items(), key=lambda kv: kv[1]):
        print(f"{name:<{col1}} | {val:>{col2}.2f}")
    print(sep)


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load backbone once ────────────────────────────────────────────────────
    full_lm, backbone, lm_head, tokenizer, eos_token_id = _load_backbone(device)

    gen_kwargs = {
        "max_length": args.max_length,
        "temp": args.temp,
        "topk": args.topk,
        "topp": args.topp,
        "eos_token_id": eos_token_id,
    }

    # ── Baseline: backbone + original lm_head ─────────────────────────────────
    baseline_file = _generate_and_save(
        label="baseline",
        backbone=backbone,
        head=lm_head,
        tokenizer=tokenizer,
        out_dir=out_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        gen_kwargs=gen_kwargs,
    )

    # ── Per-run-dir: bottleneck configs ───────────────────────────────────────
    run_dirs: list[Path] = args.run_dir or []
    adapter_files: list[tuple[str, Path]] = []

    for run_dir in run_dirs:
        ckpt_dir = run_dir / "checkpoints"
        try:
            ckpt_path = next(ckpt_dir.glob("*.pt"))
        except StopIteration:
            print(f"  WARNING: no checkpoint found in {ckpt_dir}, skipping.")
            continue

        print(f"\n[{run_dir.name}] loading adapter from {ckpt_path.name} …")
        adapter: ModelAdapter = load_adapter(ckpt_path).to(device)  # type: ignore[union-attr]
        scaler = load_scaler(run_dir / "scaler.pkl")
        head = ScaledReconHead(adapter, lm_head, scaler, device)

        out_file = _generate_and_save(
            label=run_dir.name,
            backbone=backbone,
            head=head,
            tokenizer=tokenizer,
            out_dir=out_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            gen_kwargs=gen_kwargs,
        )
        adapter_files.append((run_dir.name, out_file))

    # ── Free generation memory ────────────────────────────────────────────────
    del full_lm, backbone, lm_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Score and report ──────────────────────────────────────────────────────
    configs: list[tuple[str, Path]] = [("baseline", baseline_file), *adapter_files]
    ppls = _score_all(configs, args.ppl_model, args.max_length, out_dir)
    _print_table(ppls)


if __name__ == "__main__":
    main()

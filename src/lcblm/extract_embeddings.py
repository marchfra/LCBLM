r"""CLI to extract backbone LLM token embeddings for a HuggingFace text dataset.

Saves one .pt file per split to the output directory, compatible with
experiments/interpretability/data.py.

Output keys per file:
  input_ids       (N, L)    int64
  attention_masks (N, L)    bool
  embeddings      (N, L, D) float16
  word_ids        (N, L)    int64, -1 for special/padding tokens  [unless --no-word-ids]

The sequence length is auto-inferred as the longest tokenised sequence across
all splits (so no padding is wasted and no text is truncated).  Pass
--max-length to impose an upper cap (sequences longer than the cap are then
truncated).  With --stride S, each document is tokenised in full and sliced
into max-length-token windows that advance S tokens at a time (S == max-length
gives non-overlapping windows).

Examples
--------
# SST-2
extract-emb \\
    --model mistralai/Mistral-7B-v0.1 \\
    --dataset SetFit/sst2 \\
    --output-dir ./embeddings/sst2_mistral

# Non-overlapping 256-token windows
extract-emb \\
    --model mistralai/Mistral-7B-v0.1 \\
    --dataset SetFit/sst2 \\
    --max-length 256 --stride 256 \\
    --output-dir ./embeddings/sst2_mistral_strided

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from lcblm.utils import get_device

# ── Tokenisation ──────────────────────────────────────────────────────────────


def _infer_max_length(
    all_texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    tok_batch_size: int,
    cap: int | None,
) -> int:
    """Return the longest tokenised sequence across all texts, optionally capped."""
    max_len = 0
    for start in tqdm(
        range(0, len(all_texts), tok_batch_size),
        desc="Scanning sequence lengths",
        leave=False,
    ):
        batch = all_texts[start : start + tok_batch_size]
        enc = tokenizer(batch, truncation=False, add_special_tokens=True)
        for ids in enc["input_ids"]:
            max_len = max(max_len, len(ids))
    if cap is not None:
        max_len = min(max_len, cap)
    return max_len


def _tokenize_truncate(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    tok_batch_size: int,
    include_word_ids: bool,  # noqa: FBT001
) -> tuple[list[list[int]], list[list[int]], list[list[int]] | None]:
    """Tokenise with truncation; each text produces exactly one row."""
    all_ids: list[list[int]] = []
    all_masks: list[list[int]] = []
    all_wids: list[list[int]] | None = [] if include_word_ids else None

    for start in tqdm(
        range(0, len(texts), tok_batch_size),
        desc="Tokenising",
        leave=False,
    ):
        batch = texts[start : start + tok_batch_size]
        enc = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        all_ids.extend(enc["input_ids"])
        all_masks.extend(enc["attention_mask"])
        if all_wids is not None:
            for j in range(len(batch)):
                raw = enc.word_ids(batch_index=j)
                all_wids.append([-1 if w is None else w for w in raw])

    return all_ids, all_masks, all_wids


def _tokenize_stride(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
    *,
    include_word_ids: bool,
) -> tuple[list[list[int]], list[list[int]], list[list[int]] | None]:
    """Tokenise without truncation, then produce strided windows per text."""
    all_ids: list[list[int]] = []
    all_masks: list[list[int]] = []
    all_wids: list[list[int]] | None = [] if include_word_ids else None

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        msg = "`tokenizer` must define pad_token_id attribute"
        raise ValueError(msg)

    for text in tqdm(texts, desc="Tokenising (stride)", leave=False):
        enc = tokenizer(text, truncation=False, add_special_tokens=False)
        tokens: list[int] = enc["input_ids"]
        raw_wids: list[int | None] | None = enc.word_ids() if include_word_ids else None

        if not tokens:
            continue

        for start in range(0, len(tokens), stride):
            window = tokens[start : start + max_length]
            pad_len = max_length - len(window)
            all_ids.append(window + [pad_id] * pad_len)
            all_masks.append([1] * len(window) + [0] * pad_len)
            if all_wids is not None and raw_wids is not None:
                wslice = raw_wids[start : start + max_length]
                wrow = [-1 if w is None else w for w in wslice] + [-1] * pad_len
                all_wids.append(wrow)

    return all_ids, all_masks, all_wids


# ── Embedding extraction ──────────────────────────────────────────────────────


class _PaddedDataset(Dataset):
    def __init__(
        self,
        input_ids: list[list[int]],
        attention_masks: list[list[int]],
    ) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.attention_masks[index],
                dtype=torch.long,
            ),
        }


def _extract_embeddings(
    input_ids: list[list[int]],
    attention_masks: list[list[int]],
    llm: PreTrainedModel,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        _PaddedDataset(input_ids, attention_masks),
        batch_size=batch_size,
        shuffle=False,
    )

    collected_ids: list[torch.Tensor] = []
    collected_masks: list[torch.Tensor] = []
    collected_emb: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch in tqdm(
            loader,
            desc="Extracting embeddings",
            leave=False,
            unit="batch",
        ):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emb = llm(input_ids=ids, attention_mask=mask).last_hidden_state.cpu()
            collected_ids.append(ids.cpu())
            collected_masks.append(mask.cpu())
            collected_emb.append(emb)

    return (
        torch.cat(collected_ids, dim=0),
        torch.cat(collected_masks, dim=0).bool(),
        torch.cat(collected_emb, dim=0),
    )


# ── Split processing ──────────────────────────────────────────────────────────


def _process_split(  # noqa: PLR0913
    ds: HFDataset,
    text_column: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    llm: PreTrainedModel,
    device: torch.device,
    max_length: int | None,
    stride: int | None,
    batch_size: int,
    tok_batch_size: int,
    output_dir: Path,
    *,
    include_word_ids: bool,
) -> None:
    texts: list[str] = ds[text_column]
    print(f"\n--- {split}: {len(texts)} documents ---")

    effective_max_length = _infer_max_length(
        texts,
        tokenizer,
        tok_batch_size,
        max_length,
    )
    print(f"    max_length={effective_max_length}")

    if stride is None:
        ids, masks, wids = _tokenize_truncate(
            texts,
            tokenizer,
            effective_max_length,
            tok_batch_size,
            include_word_ids,
        )
    else:
        ids, masks, wids = _tokenize_stride(
            texts,
            tokenizer,
            effective_max_length,
            stride,
            include_word_ids=include_word_ids,
        )

    print(f"    {len(ids)} windows after tokenisation")

    t_ids, t_masks, t_emb = _extract_embeddings(ids, masks, llm, device, batch_size)

    payload: dict[str, torch.Tensor] = {
        "input_ids": t_ids,
        "attention_masks": t_masks,
        "embeddings": t_emb,
    }
    if wids is not None:
        payload["word_ids"] = torch.tensor(wids, dtype=torch.long)

    out_path = output_dir / f"extracted_features_{split}.pt"
    torch.save(payload, out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"    Saved {out_path}  shape={tuple(t_emb.shape)}  ({size_gb:.2f} GB)")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="extract-emb",
        description="Extract backbone LLM token embeddings for a HuggingFace text dataset.",  # noqa: E501
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="HuggingFace dataset name.",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory for output .pt files.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing text (default: text).",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Training split name (default: train).",
    )
    parser.add_argument(
        "--val-split",
        default="validation",
        help="Validation split name (default: validation). Skipped if absent.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=-1,
        help="Max sentences per split. -1 = all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding extraction batch size.",
    )
    parser.add_argument(
        "--tok-batch-size",
        type=int,
        default=1024,
        help="Tokenisation batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Token sequence length. "
            "Defaults to the longest sequence in the dataset (auto-inferred). "
            "Provides an upper cap when set explicitly."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Enable striding: slide a max-length-token window by this many tokens. "
            "Use --stride equal to --max-length for non-overlapping windows. "
            "Default: disabled (truncate instead)."
        ),
    )
    parser.add_argument(
        "--no-word-ids",
        action="store_true",
        help="Skip word_ids computation.",
    )

    args = parser.parse_args()

    if args.stride is not None and args.stride <= 0:
        parser.error("--stride must be a positive integer")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("HF_TOKEN") is not None:
        hf_login()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw = load_dataset(args.dataset)

    splits_to_process: dict[str, HFDataset] = {}
    for name in (args.train_split, args.val_split):
        if name in raw:
            ds: HFDataset = raw[name]
            if args.n_samples != -1:
                ds = ds.select(range(min(args.n_samples, len(ds))))
            splits_to_process[name] = ds
        elif name == args.train_split:
            parser.error(
                f"Train split '{name}' not found in dataset. Available: {list(raw)}",
            )

    for name, ds in splits_to_process.items():
        print(f"  {name}: {len(ds)} documents")

    if args.text_column not in next(iter(splits_to_process.values())).column_names:
        available = next(iter(splits_to_process.values())).column_names
        parser.error(f"Column '{args.text_column}' not found. Available: {available}")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model)  # ty:ignore[invalid-assignment]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LLM
    print(f"Loading model: {args.model}")
    device = get_device()
    print(f"Device: {device}")

    llm: PreTrainedModel = AutoModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    # Extract
    for split_name, ds in splits_to_process.items():
        _process_split(
            ds=ds,
            text_column=args.text_column,
            split=split_name,
            tokenizer=tokenizer,
            llm=llm,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.batch_size,
            tok_batch_size=args.tok_batch_size,
            include_word_ids=not args.no_word_ids,
            output_dir=output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

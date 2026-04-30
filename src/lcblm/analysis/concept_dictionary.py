"""Concept dictionary rendering — model-agnostic HTML output.

Given per-token activation values (alpha) from any concept model, produces a
self-contained HTML file with two tab-switchable views:
  - Token view: top-K token types per concept, with ±context window.
  - Sentence view: top-K sentences per concept, with peak and above-threshold
    tokens highlighted.

Background shading (amber) reflects the raw activation value; the primary token
is rendered bold blue in both views.
"""

from __future__ import annotations

import base64
import html as _html
import io
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lcblm.utils.data import NextTokenDataset

# ── CSS / JS ──────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 12px;
    color: #222;
    padding: 16px;
    background: #fff;
}
h2 { margin-bottom: 12px; font-size: 15px; color: #333; }
.summary {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.summary img { border: 1px solid #e0e0e0; border-radius: 4px; }

.grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: flex-start;
}
.col {
    flex: 0 0 auto;
    width: 230px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
}
.col-header {
    background: #f5f5f5;
    border-bottom: 1px solid #e0e0e0;
    padding: 4px 6px;
    text-align: center;
    font-weight: bold;
    font-size: 11px;
    line-height: 1.4;
}
.row {
    padding: 3px 7px;
    border-bottom: 1px solid #f0f0f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.7;
}
.row:last-child { border-bottom: none; }
.tgt { font-weight: bold; color: #1565C0; }

.sent-col {
    flex: 0 0 auto;
    width: 380px;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
}
.sent-row {
    padding: 4px 7px;
    border-bottom: 1px solid #f0f0f0;
    white-space: normal;
    word-break: break-word;
    line-height: 1.5;
    font-size: 11px;
}
.sent-row:last-child { border-bottom: none; }

.toolbar { display: flex; gap: 8px; margin-bottom: 10px; }
.tab-btn {
    padding: 5px 14px;
    border: 1px solid #bbb;
    border-radius: 4px;
    background: #f5f5f5;
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
}
.tab-btn.active { background: #1565C0; color: #fff; border-color: #1565C0; }
.tab-btn:hover:not(.active) { background: #e8e8e8; }

.legend {
    font-size: 11px;
    color: #555;
    margin-bottom: 8px;
    max-width: 900px;
    line-height: 1.6;
}
"""

_LEGEND_TOKEN = (
    "Each column shows the <strong>token types</strong> that most strongly activate "  # noqa: S105
    "this concept, ranked by their peak activation value. "
    'The <strong class="tgt">blue token</strong> is the target; surrounding gray text '
    "is its context window. Row background shading (amber) reflects the raw activation "
    "value, normalised per concept — deeper amber means stronger activation."
)
_LEGEND_WORD = (
    "Each column shows the <strong>word types</strong> that most strongly activate "
    "this concept, ranked by their peak activation value. Sub-word tokens belonging "
    'to the same word are decoded together. The <strong class="tgt">blue word</strong> '
    "is the target; surrounding gray text is its context window. Row background "
    "shading (amber) reflects the activation value, normalised per concept."
)
_LEGEND_SENTENCE = (
    "Each column shows the <strong>sentences</strong> that most strongly activate this "
    "concept, ranked by the <strong>peak activation of any single token</strong> in "
    'the sentence. The <strong class="tgt">blue token</strong> is that peak token; '
    "the amber row background reflects the sentence-level peak; "
    "other tokens above the threshold are shaded blue, with intensity "
    "proportional to their activation, normalised per concept."
)
_JS = """\
function showView(name) {
  ['token', 'sentence', 'word'].forEach(function(v) {
    var el = document.getElementById('view-' + v);
    if (el) el.style.display = v === name ? '' : 'none';
  });
  document.querySelectorAll('.tab-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.view === name);
  });
}"""


# ── Internal helpers ──────────────────────────────────────────────────────────


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _render_alpha_distributions_html(alpha: np.ndarray, threshold: float) -> str:
    """Return HTML containing a global alpha histogram and per-concept L0 bar chart."""
    n_concepts = alpha.shape[1]
    l0_per_concept = (alpha > threshold).mean(axis=0)
    order = np.argsort(l0_per_concept)[::-1]

    fig1, ax1 = plt.subplots(figsize=(4, 2.8))
    ax1.hist(alpha.ravel(), bins=50, color="#1565C0", alpha=0.75)
    ax1.axvline(
        threshold,
        color="#e53935",
        linewidth=1,
        linestyle="--",
        label=f"threshold ({threshold})",
    )
    ax1.set_xlabel("activation")
    ax1.set_ylabel("token-concept pairs")
    ax1.set_title("Global activation distribution")
    ax1.minorticks_on()
    ax1.grid(which="major", linewidth=0.6)
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    img1 = _fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(max(4, n_concepts * 0.07 + 1), 2.8))
    xs = np.arange(n_concepts)
    ax2.bar(xs, l0_per_concept[order], color="#1565C0", alpha=0.7, label="L0 freq")
    ax2.set_xlabel("concept (sorted by L0 freq)")
    ax2.set_ylabel("L0 frequency")
    ax2.set_title("Per-concept activation")
    ax2.set_xticks([])
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    img2 = _fig_to_base64(fig2)

    return (
        '<div class="summary">'
        f'<img src="data:image/png;base64,{img1}" alt="Global activation distribution">'
        f'<img src="data:image/png;base64,{img2}" alt="Per-concept activation">'
        "</div>"
    )


# ── Context retrieval ─────────────────────────────────────────────────────────


def build_context_string(
    dataset: NextTokenDataset,
    tokenizer: PreTrainedTokenizerBase,
    sentence_idx: int,
    position: int,
    context_size: int = 4,
) -> tuple[str, str, str]:
    """Return (before, target, after) strings for the token at (sentence_idx, position).

    Args:
        dataset: Dataset containing input_ids and attention_mask.
        tokenizer: Tokenizer for decoding.
        sentence_idx: Index of the sentence in the dataset.
        position: Token position within the sentence (padded index).
        context_size: Number of tokens to show on each side of the target.

    Returns:
        (before, target, after) as decoded strings. before/after include a leading
        or trailing ellipsis when the context is truncated.

    """
    sentence = dataset[sentence_idx]
    ids = sentence.input_ids.tolist()
    mask = sentence.attention_mask.tolist()

    real_positions = [i for i, m in enumerate(mask) if m]
    pos_in_real = real_positions.index(position) if position in real_positions else 0

    start = max(0, pos_in_real - context_size)
    end = min(len(real_positions), pos_in_real + context_size + 1)

    before_ids = [ids[real_positions[i]] for i in range(start, pos_in_real)]
    target_ids = [ids[position]]
    after_ids = [ids[real_positions[i]] for i in range(pos_in_real + 1, end)]

    bt_ids = before_ids + target_ids
    bta_ids = bt_ids + after_ids

    before_text = str(
        tokenizer.decode(before_ids, skip_special_tokens=True) if before_ids else "",
    )
    bt_text = str(tokenizer.decode(bt_ids, skip_special_tokens=True))
    bta_text = str(tokenizer.decode(bta_ids, skip_special_tokens=True))

    target = bt_text[len(before_text) :]
    after = bta_text[len(bt_text) :]
    before = ("… " if start > 0 else "") + before_text.lstrip()

    if end < len(real_positions):
        after = after.rstrip() + " …"

    return before, target, after


# ── Grid builders ─────────────────────────────────────────────────────────────


def _build_token_grid(  # noqa: PLR0913
    alpha: np.ndarray,
    token_ids: np.ndarray,
    sentence_indices: np.ndarray,
    positions: np.ndarray,
    dataset: NextTokenDataset,
    tokenizer: PreTrainedTokenizerBase,
    num_concepts: int,
    threshold: float,
    top_k: int,
    max_concepts: int | None,
    context_size: int,
) -> str:
    concept_total = alpha.sum(axis=0)
    top_concept_idxs = np.argsort(concept_total)[::-1][:max_concepts]

    unique_ids, inverse = np.unique(token_ids, return_inverse=True)
    n_types = len(unique_ids)
    best_occ = np.full((n_types, num_concepts), -1, dtype=np.int64)
    best_val = np.full((n_types, num_concepts), -np.inf, dtype=np.float32)
    for flat_idx in range(len(token_ids)):
        type_idx = inverse[flat_idx]
        vals = alpha[flat_idx]
        improved = vals > best_val[type_idx]
        best_val[type_idx, improved] = vals[improved]
        best_occ[type_idx, improved] = flat_idx

    parts: list[str] = ['<div class="grid">']

    for concept_idx in top_concept_idxs:
        col_alpha = alpha[:, concept_idx]
        mean_alpha = float(col_alpha.mean())
        l0_freq = float((col_alpha > threshold).mean())
        col_max = float(col_alpha.max()) or 1.0
        scores = best_val[:, concept_idx]

        valid_idxs = np.where(np.isfinite(scores))[0]
        if len(valid_idxs) == 0:
            top_type_idxs: np.ndarray = np.array([], dtype=np.int64)
        else:
            top_type_idxs = valid_idxs[np.argsort(scores[valid_idxs])[::-1][:top_k]]

        parts.append('<div class="col">')
        parts.append(
            f'<div class="col-header">C{concept_idx}'
            f"<br>L0: {l0_freq:.1%} | ā: {mean_alpha:.2f}</div>",
        )

        for type_idx in top_type_idxs:
            score = float(scores[type_idx])
            flat_idx = int(best_occ[type_idx, concept_idx])
            sent_idx = int(sentence_indices[flat_idx])
            pos = int(positions[flat_idx])

            before, target, after = build_context_string(
                dataset,
                tokenizer,
                sent_idx,
                pos,
                context_size,
            )

            bg = f"rgba(255,152,0,{score / col_max * 0.8:.3f})"
            b = _html.escape(before)
            t = _html.escape(target)
            a = _html.escape(after)
            parts.append(
                f'<div class="row" style="background:{bg}">'
                f'{b}<strong class="tgt">{t}</strong>{a}'
                "</div>",
            )

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


def _build_sentence_grid(  # noqa: PLR0913, PLR0915
    alpha: np.ndarray,
    sentence_indices: np.ndarray,
    positions: np.ndarray,
    dataset: NextTokenDataset,
    tokenizer: PreTrainedTokenizerBase,
    num_concepts: int,
    threshold: float,
    top_k: int,
    max_concepts: int | None,
) -> str:
    n_sents = dataset.num_sentences

    sent_order = np.argsort(sentence_indices, kind="stable")
    sorted_sent_idxs = sentence_indices[sent_order]
    boundaries = np.concatenate(
        [[0], np.where(np.diff(sorted_sent_idxs))[0] + 1, [len(sorted_sent_idxs)]],
    )

    best_sent_score = np.full((n_sents, num_concepts), -np.inf, dtype=np.float32)
    best_sent_flat = np.full((n_sents, num_concepts), -1, dtype=np.int64)
    sent_to_flat: dict[int, np.ndarray] = {}

    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        si = int(sorted_sent_idxs[s])
        flat_slice = sent_order[s:e]
        sent_to_flat[si] = flat_slice
        sa = alpha[flat_slice]
        best_local = np.argmax(sa, axis=0)
        best_sent_score[si] = sa[best_local, np.arange(num_concepts)]
        best_sent_flat[si] = flat_slice[best_local]

    concept_total = alpha.sum(axis=0)
    top_concept_idxs = np.argsort(concept_total)[::-1][:max_concepts]

    decoded_cache: dict[int, list[str]] = {}
    parts: list[str] = ['<div class="grid">']

    for concept_idx in top_concept_idxs:
        col_alpha = alpha[:, concept_idx]
        mean_alpha = float(col_alpha.mean())
        l0_freq = float((col_alpha > threshold).mean())
        col_max = float(col_alpha.max()) or 1.0

        scores = best_sent_score[:, concept_idx]
        valid = np.where(scores > -np.inf)[0]
        if len(valid) == 0:
            top_sent_idxs: np.ndarray = np.array([], dtype=np.int64)
        else:
            top_sent_idxs = valid[np.argsort(scores[valid])[::-1][:top_k]]

        parts.append('<div class="sent-col">')
        parts.append(
            f'<div class="col-header">C{concept_idx}'
            f"<br>L0: {l0_freq:.1%} | ā: {mean_alpha:.2f}</div>",
        )

        for sent_idx in top_sent_idxs:
            score = float(scores[sent_idx])
            best_flat = int(best_sent_flat[sent_idx, concept_idx])
            best_pos = int(positions[best_flat])

            sentence = dataset[int(sent_idx)]
            ids = sentence.input_ids.tolist()
            mask_list = sentence.attention_mask.tolist()
            real_positions = [i for i, m in enumerate(mask_list) if m]
            real_ids = [ids[p] for p in real_positions]
            best_real_local = (
                real_positions.index(best_pos) if best_pos in real_positions else 0
            )

            if int(sent_idx) not in decoded_cache:
                texts: list[str] = []
                for j in range(len(real_ids)):
                    full = str(
                        tokenizer.decode(real_ids[: j + 1], skip_special_tokens=True),
                    )
                    prev = (
                        str(tokenizer.decode(real_ids[:j], skip_special_tokens=True))
                        if j > 0
                        else ""
                    )
                    texts.append(full[len(prev) :])
                decoded_cache[int(sent_idx)] = texts
            token_texts = decoded_cache[int(sent_idx)]

            flat_slice = sent_to_flat[int(sent_idx)]
            pos_order = np.argsort(positions[flat_slice])
            sent_alpha_sorted = alpha[flat_slice[pos_order], concept_idx]

            bg = f"rgba(255,152,0,{score / col_max * 0.8:.3f})"
            row_parts: list[str] = []
            for local_i, (text, a) in enumerate(
                zip(token_texts, sent_alpha_sorted, strict=True),
            ):
                escaped = _html.escape(text)
                if local_i == best_real_local:
                    row_parts.append(f'<strong class="tgt">{escaped}</strong>')
                elif float(a) > threshold:
                    tok_bg = f"rgba(21,101,192,{float(a) / col_max * 0.35:.3f})"
                    row_parts.append(
                        f'<span style="background:{tok_bg}">{escaped}</span>',
                    )
                else:
                    row_parts.append(escaped)

            parts.append(
                f'<div class="sent-row" style="background:{bg}">'
                + "".join(row_parts)
                + "</div>",
            )

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


# ── Word view helpers ─────────────────────────────────────────────────────────


def _build_word_context_string(
    dataset: NextTokenDataset,
    tokenizer: PreTrainedTokenizerBase,
    sentence_idx: int,
    position: int,
    context_size: int = 4,
) -> tuple[str, str, str]:
    """Return (before, word, after) decoding all sub-word tokens of the target word."""
    sentence = dataset[sentence_idx]
    ids = sentence.input_ids.tolist()
    mask = sentence.attention_mask.tolist()
    real_positions = [i for i, m in enumerate(mask) if m]

    if sentence.word_ids is None:
        return build_context_string(
            dataset,
            tokenizer,
            sentence_idx,
            position,
            context_size,
        )

    word_ids_list = sentence.word_ids.tolist()
    target_word_id = word_ids_list[position]

    word_pos = [p for p in real_positions if word_ids_list[p] == target_word_id]
    if not word_pos:
        return build_context_string(
            dataset,
            tokenizer,
            sentence_idx,
            position,
            context_size,
        )

    first_in_real = real_positions.index(word_pos[0])
    last_in_real = real_positions.index(word_pos[-1])

    start = max(0, first_in_real - context_size)
    end = min(len(real_positions), last_in_real + context_size + 1)

    before_ids = [ids[real_positions[i]] for i in range(start, first_in_real)]
    target_ids = [ids[p] for p in word_pos]
    after_ids = [ids[real_positions[i]] for i in range(last_in_real + 1, end)]

    bt_ids = before_ids + target_ids
    bta_ids = bt_ids + after_ids

    before_text = (
        str(tokenizer.decode(before_ids, skip_special_tokens=True))
        if before_ids
        else ""
    )
    bt_text = str(tokenizer.decode(bt_ids, skip_special_tokens=True))
    bta_text = str(tokenizer.decode(bta_ids, skip_special_tokens=True))

    target_text = bt_text[len(before_text) :]
    after_text = bta_text[len(bt_text) :]

    before = ("… " if start > 0 else "") + before_text.lstrip()
    after = after_text
    if end < len(real_positions):
        after = after.rstrip() + " …"

    return before, target_text, after


def _build_word_grid(  # noqa: PLR0913
    word_alpha: np.ndarray,
    word_token_ids: np.ndarray,
    word_sentence_indices: np.ndarray,
    word_positions: np.ndarray,
    dataset: NextTokenDataset,
    tokenizer: PreTrainedTokenizerBase,
    num_concepts: int,
    threshold: float,
    top_k: int,
    max_concepts: int | None,
    context_size: int,
) -> str:
    concept_total = word_alpha.sum(axis=0)
    top_concept_idxs = np.argsort(concept_total)[::-1][:max_concepts]

    unique_ids, inverse = np.unique(word_token_ids, return_inverse=True)
    n_types = len(unique_ids)
    best_occ = np.full((n_types, num_concepts), -1, dtype=np.int64)
    best_val = np.full((n_types, num_concepts), -np.inf, dtype=np.float32)
    for flat_idx in range(len(word_token_ids)):
        type_idx = inverse[flat_idx]
        vals = word_alpha[flat_idx]
        improved = vals > best_val[type_idx]
        best_val[type_idx, improved] = vals[improved]
        best_occ[type_idx, improved] = flat_idx

    parts: list[str] = ['<div class="grid">']

    for concept_idx in top_concept_idxs:
        col_alpha = word_alpha[:, concept_idx]
        mean_alpha = float(col_alpha.mean())
        l0_freq = float((col_alpha > threshold).mean())
        col_max = float(col_alpha.max()) or 1.0
        scores = best_val[:, concept_idx]

        valid_idxs = np.where(np.isfinite(scores))[0]
        if len(valid_idxs) == 0:
            top_type_idxs: np.ndarray = np.array([], dtype=np.int64)
        else:
            top_type_idxs = valid_idxs[np.argsort(scores[valid_idxs])[::-1][:top_k]]

        parts.append('<div class="col">')
        parts.append(
            f'<div class="col-header">C{concept_idx}'
            f"<br>L0: {l0_freq:.1%} | ā: {mean_alpha:.2f}</div>",
        )

        for type_idx in top_type_idxs:
            score = float(scores[type_idx])
            flat_idx = int(best_occ[type_idx, concept_idx])
            sent_idx = int(word_sentence_indices[flat_idx])
            pos = int(word_positions[flat_idx])

            before, target, after = _build_word_context_string(
                dataset,
                tokenizer,
                sent_idx,
                pos,
                context_size,
            )

            bg = f"rgba(255,152,0,{score / col_max * 0.8:.3f})"
            b = _html.escape(before)
            t = _html.escape(target)
            a = _html.escape(after)
            parts.append(
                f'<div class="row" style="background:{bg}">'
                f'{b}<strong class="tgt">{t}</strong>{a}'
                "</div>",
            )

        parts.append("</div>")

    parts.append("</div>")
    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────


def build_concept_dictionary(  # noqa: PLR0913
    alpha: np.ndarray,
    token_ids: np.ndarray,
    sentence_indices: np.ndarray,
    positions: np.ndarray,
    dataset: NextTokenDataset,
    tokenizer: PreTrainedTokenizerBase,
    num_concepts: int,
    threshold: float,
    top_k: int = 12,
    max_concepts: int | None = None,
    context_size: int = 4,
    title: str = "Concept Dictionary",
    out_path: Path | None = None,
    word_alpha: np.ndarray | None = None,
    word_token_ids: np.ndarray | None = None,
    word_sentence_indices: np.ndarray | None = None,
    word_positions: np.ndarray | None = None,
) -> None:
    """Write a self-contained HTML concept dictionary.

    Produces a token view, a sentence view, and — when word arrays are supplied —
    a word view. All views are tab-switchable inside a single HTML file.

    Args:
        alpha: Activation values, shape (N_tokens, num_concepts). For VAEE this is gate
            probability in [0, 1]; for SAE this is the post-activation latent value.
        token_ids: Flat token IDs, shape (N_tokens,).
        sentence_indices: Sentence index for each token, shape (N_tokens,).
        positions: Position within the padded sequence for each token, shape
            (N_tokens,).
        dataset: Dataset used to retrieve context tokens and full sentences.
        tokenizer: HuggingFace tokenizer for decoding.
        num_concepts: Total number of concepts in the model.
        threshold: Activation value above which a concept is considered active for L0
            frequency computation. Use 0.5 for VAEE gate probabilities, 0.0 for SAE.
        top_k: Number of entries to display per concept per view.
        max_concepts: Maximum number of concepts to show. None shows all.
        context_size: Tokens to show on each side of the target token or word.
        title: Page heading.
        out_path: Path to write the HTML file. Defaults to concept_dict.html.
        word_alpha: Word-level activations from aggregate_to_words, shape
            (N_words, num_concepts). Required for the word view tab.
        word_token_ids: First sub-word token ID for each word, shape (N_words,).
        word_sentence_indices: Sentence index for each word, shape (N_words,).
        word_positions: Position of the first sub-word token of each word,
            shape (N_words,).

    """
    if out_path is None:
        out_path = Path("concept_dict.html")

    has_word_view = all(
        x is not None
        for x in (word_alpha, word_token_ids, word_sentence_indices, word_positions)
    )

    token_grid = _build_token_grid(
        alpha,
        token_ids,
        sentence_indices,
        positions,
        dataset,
        tokenizer,
        num_concepts,
        threshold,
        top_k,
        max_concepts,
        context_size,
    )
    sentence_grid = _build_sentence_grid(
        alpha,
        sentence_indices,
        positions,
        dataset,
        tokenizer,
        num_concepts,
        threshold,
        top_k,
        max_concepts,
    )
    if has_word_view:
        word_grid = _build_word_grid(
            word_alpha,  # ty:ignore[invalid-argument-type]
            word_token_ids,  # ty:ignore[invalid-argument-type]
            word_sentence_indices,  # ty:ignore[invalid-argument-type]
            word_positions,  # ty:ignore[invalid-argument-type]
            dataset,
            tokenizer,
            num_concepts,
            threshold,
            top_k,
            max_concepts,
            context_size,
        )

    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html lang='en'><head>",
        "<meta charset='UTF-8'>",
        f"<title>{_html.escape(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        f"<h2>{_html.escape(title)}</h2>",
        _render_alpha_distributions_html(alpha, threshold=threshold),
        '<div class="toolbar">',
        '<button class="tab-btn active" data-view="token" onclick="showView(\'token\')">Token view</button>',  # noqa: E501
        '<button class="tab-btn" data-view="sentence" onclick="showView(\'sentence\')">Sentence view</button>',  # noqa: E501
    ]
    if has_word_view:
        parts.append(
            '<button class="tab-btn" data-view="word" onclick="showView(\'word\')">Word view</button>',  # noqa: E501
        )
    parts += [
        "</div>",
        '<div id="view-token">',
        f'<p class="legend">{_LEGEND_TOKEN}</p>',
        token_grid,
        "</div>",
        '<div id="view-sentence" style="display:none">',
        f'<p class="legend">{_LEGEND_SENTENCE}</p>',
        sentence_grid,
        "</div>",
    ]
    if has_word_view:
        parts += [
            '<div id="view-word" style="display:none">',
            f'<p class="legend">{_LEGEND_WORD}</p>',
            word_grid,  # type: ignore[name-defined]
            "</div>",
        ]
    parts += [
        f"<script>{_JS}</script>",
        "</body></html>",
    ]

    html_str = "\n".join(parts)
    out_path = Path(out_path).with_suffix(".html")
    out_path.write_text(html_str, encoding="utf-8")
    print(f"Saved {out_path}")

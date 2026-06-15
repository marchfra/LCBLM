"""Regenerate report.html for the dict-learning paper from sweep outputs.

Builds tables directly from each tier's results.json (no hand-transcription) and
embeds the 2D-analysis + learning-curve figures as base64 (self-contained HTML).

Usage:
    python experiments/dict_learning_paper/make_report.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

# Bump this for each major report revision; the versioned file is archived and
# report.html is updated as the "latest" copy. (v1 was an untracked hand-written
# report that predates this generator.)
VERSION = "v2"
_DATE = "2026-06-09"

_ROOT = Path(__file__).parent
_OUT = _ROOT / f"report_{VERSION}.html"
_LATEST = _ROOT / "report.html"

# Latest 2D-tier sweep dirs (low-dim validation).
_TIERS = [
    (
        "easy",
        "Easy — 8 atoms, k=1 singletons",
        "2D, 8 atoms on the unit circle, exactly one active atom per sample, "
        "binary coefficients, σ=0.05. Pure single-atom recovery — every model "
        "should pass. (Run uses beta=1 for the VAEE family — beta=4, the "
        "VAEE-SE-tuned value, collapses standard-VAEE recovery; see findings.)",
        _ROOT / "outputs/sweep_synthetic_easy/20260528-233244",
    ),
    (
        "medium",
        "Medium — 5 atoms, binary superposition",
        "2D, 5 atoms, independent Bernoulli activation (p=0.5) with the "
        "adjacent-only constraint (≈50% singletons + 50% ring-adjacent pairs), "
        "binary coefficients, σ=0.05. Tests decomposition of sums into atoms.",
        _ROOT / "outputs/sweep_synthetic_medium/20260608-142248",
    ),
    (
        "hard",
        "Hard — 5 atoms, continuous superposition",
        "Same support recipe as medium, but continuous per-sample magnitudes "
        "cᵢ ~ Uniform(0.5, 1.5). Only a per-sample encoder can reconstruct the "
        "varying magnitude exactly; a fixed prototype hits a magnitude floor.",
        _ROOT / "outputs/sweep_synthetic_hard/20260608-165009",
    ),
]

_MODEL_ORDER = [
    "vaee",
    "vaee_shared_encoder",
    "topk_sae",
    "sae_concept",
    "vq_vae",
    "beta_vae",
]
_MODEL_LABEL = {
    "vaee": "VAEE",
    "vaee_shared_encoder": "VAEE-SE",
    "topk_sae": "TopK-SAE",
    "sae_concept": "L1-SAE",
    "vq_vae": "VQ-VAE",
    "beta_vae": "β-VAE",
}


def _b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def _rows(run_dir: Path) -> list[dict]:
    out = []
    for p in sorted(run_dir.glob("*/results.json")):
        d = json.loads(p.read_text())
        model = d.get("model_name", "")
        out.append(
            {
                "model": model,
                "cfg": p.parent.name[len(model) + 1 :],
                "matched": d.get("matched_fraction"),
                "cos": d.get("mean_cosine_sim"),
                "mse": d.get("best_val_recon"),
                "active": d.get("best_l0"),
                "alive": d.get("alive_dict_size"),
            }
        )
    out.sort(
        key=lambda r: (
            _MODEL_ORDER.index(r["model"]) if r["model"] in _MODEL_ORDER else 99,
            r["cfg"],
        )
    )
    return out


def _fmt(v: float | None, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def _table(rows: list[dict]) -> str:
    best = max((r["matched"] or -1) for r in rows) if rows else None
    head = (
        "<table><thead><tr><th>model</th><th>config</th><th>matched&nbsp;↑</th>"
        "<th>cos&nbsp;↑</th><th>MSE&nbsp;↓</th><th>#active&nbsp;concepts</th>"
        "<th>alive</th></tr></thead><tbody>"
    )
    body = []
    for r in rows:
        cls = []
        if r["model"] == "vaee":
            cls.append("vaee")
        if r["matched"] is not None and r["matched"] == best and best and best > 0:
            cls.append("best")
        c = f" class='{' '.join(cls)}'" if cls else ""
        body.append(
            f"<tr{c}><td class='m'>{_MODEL_LABEL.get(r['model'], r['model'])}</td>"
            f"<td class='cfg'>{r['cfg']}</td><td>{_fmt(r['matched'], 2)}</td>"
            f"<td>{_fmt(r['cos'])}</td><td>{_fmt(r['mse'], 4)}</td>"
            f"<td>{_fmt(r['active'], 2)}</td><td>{r['alive'] if r['alive'] is not None else '—'}</td></tr>"
        )
    return head + "".join(body) + "</tbody></table>"


def _figure(path: Path, caption: str) -> str:
    b = _b64(path)
    if b is None:
        return f"<p class='legend'>[missing figure: {path.name}]</p>"
    return (
        f"<figure><img src='data:image/png;base64,{b}'/>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


_CSS = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;max-width:960px;margin:2.2rem auto;padding:0 1.2rem;background:#fff}
h1{font-size:1.7rem;margin:0 0 .2rem} h2{font-size:1.25rem;margin:2rem 0 .6rem;border-bottom:2px solid #eee;padding-bottom:.25rem}
h3{font-size:1.05rem;margin:1.3rem 0 .4rem;color:#333}
.sub{color:#666;margin:0 0 1.2rem}
.callout{background:#eef6ff;border-left:4px solid #3478d3;padding:.7rem 1rem;border-radius:4px;margin:1rem 0}
.warn{background:#fff8ef;border-left:4px solid #d38a34;padding:.7rem 1rem;border-radius:4px;margin:1rem 0}
.note{background:#f6f8fa;border-left:4px solid #bbb;padding:.6rem 1rem;border-radius:4px;margin:1rem 0;font-size:.93rem}
table{border-collapse:collapse;width:100%;margin:.7rem 0 1.1rem;font-size:.9rem}
caption{caption-side:top;text-align:left;font-weight:600;color:#444;padding:.3rem 0;font-size:.92rem}
th,td{border:1px solid #e2e2e2;padding:.35rem .55rem;text-align:right}
th{background:#f4f4f4;font-weight:600} td.m,td.cfg{text-align:left}
td.m{font-weight:600}
tr.vaee td{background:#fff0f0} tr.best td{background:#eefaf0}
ul{margin:.4rem 0 .9rem;padding-left:1.3rem} li{margin:.25rem 0}
figure{margin:1rem 0;text-align:center} img{max-width:100%;border:1px solid #e2e2e2;border-radius:6px}
figcaption{color:#666;font-size:.85rem;margin-top:.35rem}
code{background:#f0f0f0;padding:.05rem .3rem;border-radius:3px;font-size:.88em}
.legend{color:#666;font-size:.85rem;margin:-.3rem 0 1rem}
"""

_INTRO = """
<h1>VAEE as Dictionary Learning — Intermediate Report</h1>
<p class='sub'>Workshop-paper track · {version} · phase: synthetic feature-recovery validation (2D tiers) · {date}</p>
<p>We validate VAEE as a dictionary-learning method on controlled 2D synthetic data with <b>known ground-truth atoms</b>, so recovery is measured directly: Hungarian matching of learned prototype directions to true atoms, reported as <b>matched fraction</b> (atoms recovered above cos 0.9) and <b>mean cosine similarity</b>. Three tiers form an escalating ladder defined by the <b>coefficient / support structure</b>: single atoms (easy) → binary superposition (medium) → continuous-magnitude superposition (hard). High-dimensional versions of all three are running separately.</p>
<div class='note'><b>Reading the metrics.</b> <b>#active concepts</b> is the per-sample count of firing concepts — the sparsity axis (the headline Pareto is MSE vs #active concepts, <i>not</i> raw L0). For VAEE a concept is an <b>E-dimensional subspace</b> (prototype + per-sample variation), so its true capacity is #active × E; an SAE latent is a 1-D ray (direction × scalar). Matched #active is therefore <i>not</i> matched capacity — a point we keep explicit when comparing MSE.</div>
"""

_FINDINGS = """
<h2>Findings so far (2D)</h2>

<div class='callout'><b>Headline.</b> Standard VAEE is the only model whose recovery <b>improves as the task gets richer</b> — it fails the trivial tier but nails the hardest. On <b>hard</b> (continuous magnitudes) it is the only model that recovers atoms <i>and</i> reconstructs them well; SAEs reconstruct (trivially, in 2D) on a non-atomic basis, and VQ-VAE / β-VAE break.</div>

<h3>Standard VAEE improves with difficulty — but it's a coverage story, not a ceiling</h3>
<ul>
<li>Recovery climbs <b>easy ≤0.5 → medium 0.80 → hard 1.00</b> (best operating point), the inverse of every other model.</li>
<li>The easy failure is <b>not</b> a hyperparameter artifact: across ~15 prior runs (β, γ, λ_ent, λ_ortho sweeps) standard VAEE never beat <b>0.375</b> on the k=1 tier. The 2D plots show the mechanism — a <b>prototype-coverage failure</b>: some atoms get two prototypes, others get none ([1,1,1,2,0] instead of [1,1,1,1,1]). Nothing in the objective penalizes this: the orthogonality term acts in embedding space and is geometrically capped (you can't have 5 mutually-orthogonal directions in 2D), so there is no repulsion to spread prototypes onto all atoms. Coverage is left to initialization, hence seed-dependent.</li>
<li><b>Caveat:</b> with only 5–8 atoms, matched fraction is quantized in steps of 0.125–0.2 and single-seed runs land in different coverage minima, so individual numbers carry a ±0.2–0.4 noise band. Multi-seed runs are required before any of these are stated as stable — flagged as a hard requirement for the camera-ready.</li>
</ul>

<h3>The per-sample encoder is the differentiator on continuous data</h3>
<ul>
<li>On binary tiers VAEE and VAEE-SE tie on recovery. On <b>hard</b> they both recover (matched 1.0) but separate on reconstruction: <b>VAEE MSE 0.010 vs VAEE-SE 0.046</b>. VAEE-SE's fixed prototype reconstructs (mean magnitude)·atom — a magnitude floor — while VAEE's per-sample μ(x) encodes the actual magnitude (the apple is "red", and μ carries <i>this</i> shade).</li>
<li>μ at inference is the free posterior mean (not sampled), so each active concept spans an E-dim subspace around its prototype. Its freedom to fit per-sample residual is <b>rate-limited by the KL</b> (β), not by capacity: a β-sweep shows MSE rising ~30× and #active concepts falling as β grows, and at β=1 VAEE drives MSE to ≈0.002. So VAEE <i>can</i> fit residual via μ; β sets how much.</li>
</ul>

<h3>Why the SAEs' near-zero MSE is not a quality signal (in 2D)</h3>
<ul>
<li>TopK/L1 reach MSE 10⁻⁴–10⁻⁷ but matched only 0.4–0.6. In 2D, two continuous active units span the whole plane, so the decoder fits <i>any</i> point — including the noise — on a non-atomic basis. MSE is degenerate here; recovery is the only honest axis.</li>
<li>VAEE's higher MSE is <b>regularization-limited (KL), not capacity-limited</b> — it has the parameters to overfit but the variational prior makes fitting noise cost more than it saves. <b>This is exactly why the high-dim tiers matter</b>: with k≈3 active in 32-D, k units cannot span the space, no model can noise-fit, and MSE becomes a fair recovery proxy.</li>
</ul>

<h3>Baselines</h3>
<ul>
<li><b>VQ-VAE</b> recovers only by degeneracy: a fair codebook (size = #atoms) fails (matched 0.40, high MSE); it reaches 1.0 only by spending extra codes to memorise the discrete sample modes — and on continuous (hard) data even that degrades.</li>
<li><b>SAEs</b> plateau at matched 0.4–0.6 (recover ~2 atoms, reconstruct the rest via mixtures).</li>
<li><b>β-VAE</b> suffers posterior collapse (1–3 alive latents) and the worst MSE.</li>
<li>Overcomplete VAEE (K&gt;atoms) self-prunes back toward the true atom count via its gate-KL (e.g. K=20→~5 alive) — encouraging for the "overcomplete + prune rare latents" direction, though it does not by itself fix the easy-tier coverage problem.</li>
</ul>
"""

_PENDING = """
<h2>In progress / pending</h2>
<ul>
<li><b>High-dimensional tiers</b> (D=32, K=64 atoms = 2× overcomplete, E[k]=3, small noise) for all three recipes — running. These are the fair, non-degenerate MSE comparison and the test of whether the 2D rankings generalise.</li>
<li><b>Multi-seed</b> (≥5) on the key cells for mean ± std — required before any matched-fraction claim is camera-ready.</li>
<li><b>Coverage fix</b> for standard VAEE on trivial data (data-driven prototype init and/or an input-space repulsion term) — diagnosed, not yet implemented.</li>
<li>Image tiers (MNIST / Fashion-MNIST / dSprites) and the cross-dataset L0–MSE Pareto figure.</li>
</ul>
"""


def build() -> str:
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>VAEE as Dictionary Learning — Intermediate Report ({VERSION})</title><style>{_CSS}</style></head><body>"
    ]
    parts.append(_INTRO.format(version=VERSION, date=_DATE))
    parts.append("<h2>Results — 2D tiers</h2>")
    for _key, title, desc, run_dir in _TIERS:
        parts.append(f"<h3>{title}</h3>")
        parts.append(f"<p class='legend'>{desc}</p>")
        parts.append(_table(_rows(run_dir)))
        parts.append(
            _figure(
                run_dir / "2d_analysis.png",
                "Learned prototype directions (arrows) over the validation scatter, best operating point per model.",
            )
        )
        parts.append(
            _figure(
                run_dir / "learning_curves.png",
                "Per-sweep learning curves: recon MSE (log) and # active concepts, shared scales across models.",
            )
        )
    parts.append(_FINDINGS)
    parts.append(_PENDING)
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    html = build()
    _OUT.write_text(html)  # archived versioned file
    _LATEST.write_text(html)  # "latest" pointer
    size_kb = _OUT.stat().st_size / 1024
    print(f"Wrote {_OUT.name} + {_LATEST.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

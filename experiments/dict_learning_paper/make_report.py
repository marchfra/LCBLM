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
VERSION = "v4"
_DATE = "2026-06-17"

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

# High-dim tiers (D=32, K=64 atoms = 2x overcomplete, E[k]=3) — the fair,
# non-degenerate MSE arbiter: with k active in 32-D, k units cannot span the
# space, so no model can trivially noise-fit (unlike 2D).
_HIGHDIM_TIERS = [
    (
        "highdim_easy",
        "High-dim easy — 64 atoms, k=1 singletons",
        "D=32, K=64 atoms, exactly one active atom per sample, σ=0.05. The "
        "high-dim analogue of the 2D easy tier; recovery threshold 0.7 "
        "(32-D random pairs sit at cos≈0.18, so 0.9 is too strict).",
        _ROOT / "outputs/sweep_synthetic_highdim_easy/20260610-115057",
    ),
    (
        "highdim_medium",
        "High-dim medium — binary superposition",
        "D=32, K=64, Bernoulli active_prob=0.047 (E[k]≈3), binary coefficients, "
        "σ=0.05. The first tier where k≪D, so MSE is a real recovery proxy.",
        _ROOT / "outputs/sweep_synthetic_highdim_medium/20260610-155819",
    ),
    (
        "highdim_hard",
        "High-dim hard — continuous superposition",
        "Same support as medium, continuous per-sample magnitudes "
        "cᵢ~Uniform(0.5,1.5), σ=0.05. VAEE rows here are the resample-OFF "
        "baseline; resampling lifts VAEE to matched≈0.53–0.72 (see the "
        "within-concept-variation section, σ=0.05 panel).",
        _ROOT / "outputs/sweep_synthetic_highdim_hard/20260610-163827",
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
<p>We validate VAEE as a dictionary-learning method on controlled 2D synthetic data with <b>known ground-truth atoms</b>, so recovery is measured directly: Hungarian matching of learned prototype directions to true atoms, reported as <b>matched fraction</b> (atoms recovered above cos 0.9) and <b>mean cosine similarity</b>. Three tiers form an escalating ladder defined by the <b>coefficient / support structure</b>: single atoms (easy) → binary superposition (medium) → continuous-magnitude superposition (hard). We then repeat the ladder in <b>high dimension</b> (D=32, K=64 atoms = 2× overcomplete, E[k]≈3) — the regime where MSE becomes a fair recovery proxy — and sweep <b>within-concept variation</b> (additive noise σ=0.05/0.10/0.20) on the hardest high-dim tier.</p>
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

_HIGHDIM_FINDINGS = """
<h2>Findings — high-dim tiers (D=32, K=64)</h2>

<div class='warn'><b>Honest headline (this inverts the 2D story).</b> In the fair, non-degenerate regime, <b>TopK-SAE is the robust recovery winner</b>: matched 0.92–0.98 at exactly <i>k</i> active concepts, on every tier. VAEE's 2D recovery advantage <b>does not carry over</b> — overcomplete VAEE is capped by latent/concept <b>death</b> (alive 18–44 of 64), and even with dead-latent resampling (alive→60–62) it reaches only matched≈0.53–0.72, below TopK. We report this plainly rather than cherry-picking the axis where VAEE leads.</div>

<ul>
<li><b>TopK-SAE</b> recovers cleanly and cheaply: matched ≈0.95 at #active = k (2–5), MSE degrading gracefully with k. It is the baseline to beat in high-dim and it is currently winning recovery.</li>
<li><b>L1-SAE</b> is the <b>degenerate noise-fitter</b>: it posts the lowest MSE (10⁻³–10⁻⁴) but with a <i>dense</i> code (#active ≈ 27–33, half the dictionary) and poor recovery — though raising λ (0.3) trades MSE for recovery and reaches matched 0.84 on hard. Low MSE alone is not interpretability.</li>
<li><b>VAEE</b> reaches matched 0.42–0.48 at the resample-off baseline, lifted to 0.53–0.72 by resampling (see next section). Its MSE (≈0.012) is competitive with TopK at the same operating point, but at more active concepts and lower recovery — so it does <i>not</i> dominate TopK on any single axis here.</li>
<li><b>VAEE-SE</b> fails in high-dim continuous superposition (matched 0, cos≈0.29 ≈ the 32-D random-pair floor): a fixed prototype cannot absorb continuous per-sample magnitudes, so it spreads activation across many concepts and never locks onto atoms.</li>
<li><b>VQ-VAE</b> and <b>β-VAE</b> both break in high-dim (matched ≈0; β-VAE collapses to 0–1 alive latents).</li>
</ul>
<div class='note'><b>Single-seed caveat (unchanged, and load-bearing here).</b> Every high-dim number is one seed; the recovery band is ±0.2–0.4. The qualitative ranking (TopK ≫ VAEE &gt; VAEE-SE/VQ/β on recovery) is stable across configs, but the specific VAEE matched fractions are not camera-ready until the ≥3-seed reruns land.</div>
"""

_NOISE_FINDINGS = """
<h2>Within-concept variation — does VAEE's per-concept lead widen with noise?</h2>
<p class='legend'>High-dim hard tier re-run at three additive-noise levels (σ = 0.05 / 0.10 / 0.20) with dead-latent resampling on. Additive noise is our (weak) proxy for intra-concept variation: samples sharing a concept support are no longer identical. The question was whether VAEE's reconstruction-per-active-concept advantage <i>widens</i> as within-concept variation grows.</p>

<div class='warn'><b>Answer: no — the hypothesis is not supported.</b> As σ grows, VAEE's recovery <b>collapses</b> (matched 0.67 → 0.38 → 0.00) and its #active concepts <b>inflate</b> (7 → 14), drifting toward L1's dense failure mode. <b>TopK-SAE stays robust</b> (matched 0.91–0.98 at k active throughout). VAEE is competitive only in the low-noise window; it does not pull ahead under within-concept variation, it falls behind.</div>

<ul>
<li><b>L1-SAE's MSE <i>decreases</i> as noise rises</b> (0.0005 → 0.0001) — the tell-tale signature of noise-fitting with a dense code; recovery stays 0.08–0.84 (λ-dependent). Not a real competitor on recovery.</li>
<li>The Pareto figure makes the multi-axis picture explicit: <b>TopK owns the upper-left corner</b> (few active concepts + high recovery) at every noise level. The "fewer concepts to inspect for a given reconstruction" claim does not hold here — at σ≥0.10 VAEE uses <i>more</i> active concepts than TopK, not fewer.</li>
<li><b>Why additive noise is the wrong test.</b> Isotropic example-level noise is structureless: there is no shared low-rank direction for VAEE's per-sample μ to exploit, so μ can only fit it by burning rate (KL) or by recruiting extra concepts — exactly the degradation observed. The principled test is <b>structured</b> intra-concept variance (per-concept low-rank subspaces, x = Σ gateᵢ·(aᵢ + Bᵢwᵢ)), deferred to a <code>make_complex_synthetic</code> generator. The current result should be read as "additive noise hurts VAEE", not "VAEE has no within-concept-variation advantage" — the latter is untested.</li>
</ul>
"""

_EMB_FINDINGS = """
<h2>Embedding size E is a capacity lever — and it rescues VAEE-SE</h2>
<p class='legend'>High-dim hard tier (σ=0.05), π fixed at 0.047, resampling on. Single seed — directional, not camera-ready. Capacity = #active × E.</p>

<div class='callout'><b>The E=4 high-dim collapse was a gating-capacity artifact, not architectural.</b> With K=64 prototypes crammed into a 4-D embedding space the gates cannot separate, so concepts never lock onto atoms. Enlarging E relieves this for <i>both</i> variants — and <b>VAEE-SE goes from matched 0.00 (E=4) to 0.94 (E=64)</b>, becoming the best recoverer of the family.</div>

<table>
<caption>Recovery (matched ↑) and reconstruction (MSE ↓) vs embedding size E</caption>
<thead><tr><th>E</th><th>VAEE matched</th><th>VAEE MSE</th><th>VAEE-SE matched</th><th>VAEE-SE MSE</th></tr></thead>
<tbody>
<tr><td>4</td><td>0.53</td><td>0.0123</td><td>0.00</td><td>0.083</td></tr>
<tr><td>8</td><td>0.56</td><td>0.0110</td><td>0.38</td><td>0.059</td></tr>
<tr><td>16</td><td>0.66</td><td>0.0101</td><td>0.75</td><td>0.034</td></tr>
<tr><td>32</td><td>0.70</td><td>0.0084</td><td>0.88</td><td>0.021</td></tr>
<tr><td>64</td><td>0.83</td><td>0.0076</td><td>0.94</td><td>0.017</td></tr>
<tr><td>128</td><td>0.88</td><td>0.0071</td><td>0.92</td><td>0.017</td></tr>
</tbody></table>

<ul>
<li><b>Both saturate around E≈64</b> (VAEE 0.83→0.88, VAEE-SE 0.94→0.92, the dip within single-seed noise). TopK reference: matched 0.98, MSE 0.0127.</li>
<li><b>Clean variant split.</b> VAEE-SE (fixed prototype, no per-sample μ) is the better <b>recoverer</b> (0.92–0.94, near TopK's 0.98) — it cannot fit per-sample magnitude, so it stays glued to atom directions; VAEE (per-sample μ) is the better <b>reconstructor</b> (MSE 0.0071, beating TopK) but blurs atoms via μ.</li>
<li><b>The honest cost:</b> this near-parity costs ≈90× the capacity (E=128 ⇒ ~460 vs TopK's 5), on a dataset that <i>is</i> the linear-sparse SAE generative model. So it is expensive parity, not a win — the "loses badly to TopK" impression is an E=4 artifact, but the converse (a real VAEE win) this tier cannot produce.</li>
</ul>
"""

_INTERVENTION_FINDINGS = """
<h2>Intervention (PLAN.md metrics 6 & 7) — and why this tier can't settle it</h2>
<p class='legend'>Ablation-based, model-agnostic (<code>eval/intervention.py</code>): for each active concept, zero its activation, decode, and measure the induced Δ. <b>Consistency</b> = mean resultant length of the unit Δ-directions across inputs (metric 6); <b>causal matched / dominance</b> = Hungarian match of mean Δ-directions to atoms + top/second-atom ratio (metric 7).</p>

<table>
<caption>Intervention metrics, high-dim hard σ=0.05 (best E per VAEE variant; SAE at E=4)</caption>
<thead><tr><th>model</th><th>consistency&nbsp;↑</th><th>causal matched&nbsp;↑</th><th>dominance&nbsp;↑</th></tr></thead>
<tbody>
<tr class='best'><td class='m'>TopK-SAE (k=5)</td><td>1.00</td><td>0.98</td><td>2.36</td></tr>
<tr class='vaee'><td class='m'>VAEE (E=128)</td><td>0.99</td><td>0.89</td><td>2.22</td></tr>
<tr><td class='m'>VAEE-SE (E=64)</td><td>1.00</td><td>0.92</td><td>2.12</td></tr>
<tr><td class='m'>L1-SAE (λ=0.3)</td><td>1.00</td><td>0.84</td><td>1.59</td></tr>
</tbody></table>

<div class='warn'><b>The intervention metrics are correctly implemented but cannot reveal a VAEE advantage on this data — by construction.</b> Two structural reasons:
<ul>
<li><b>Consistency is degenerate on a linear decoder.</b> Ablating an SAE / VAEE-SE concept removes its fixed decoder column, input-independent ⇒ consistency = 1.000 trivially. VAEE is the <i>only</i> model below 1.0 (0.98), because its per-sample μ makes the direction wander — so the variational machinery slightly <i>hurts</i> the metric the plan expected it to win. Consistency only becomes informative with a <b>nonlinear</b> decoder (the plan's ResNet-feature-space metric 6), invisible here.</li>
<li><b>The metric reduces each concept to a single direction</b> — the SAE-native object. It rewards 1-D concepts and is blind to VAEE's E-dim subspace. Causal-matched simply tracks prototype recovery for every model (the metric is self-consistent, not buggy), so the gap to TopK is just the recovery gap, which is the dataset.</li>
</ul>
VAEE's only edge here is <b>dominance</b> (interventions concentrate on one atom), and even that trails TopK. Verdict: on the linear-sparse tier, intervention favors the model whose inductive bias matches the generator (SAE). The variational-controllability claim must be tested where it can show — structured subspaces or nonlinear features.</div>
"""

_PENDING = """
<h2>In progress / pending</h2>
<ul>
<li><b>Structured intra-concept variance</b>: <code>make_complex_synthetic</code> with per-concept low-rank subspaces (mixture-of-factor-analyzers, x = Σ gateᵢ·(aᵢ + Bᵢwᵢ)). <b>Now the priority and the blocker</b>: every result above is on the linear-sparse tier that <i>is</i> the SAE generative model, so it cannot produce a VAEE win. Only structured subspaces give VAEE's E-dim concept something a 1-D SAE latent cannot represent — the regime where recovery, MSE and intervention can all favor VAEE. A 2D-visualisable version is planned.</li>
<li><b>Multi-seed</b> (≥3, ideally 5) on the key cells for mean ± std — the hard gate before any matched-fraction goes camera-ready. Decision taken: <b>fix π = E[k]/K and sweep seed instead of π</b>. Every E-sweep / noise / intervention number above is single-seed.</li>
<li><b>Nonlinear intervention</b>: metric 6 in ResNet / LLM feature space, where the consistency metric is no longer degenerate and the variational posterior can pay off.</li>
<li><b>Coverage fix</b> for standard VAEE on trivial 2D data (data-driven prototype init and/or an input-space repulsion term) — diagnosed, not yet implemented.</li>
<li>Image tiers (MNIST / Fashion-MNIST / dSprites) and the cross-dataset Pareto figure.</li>
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

    parts.append("<h2>Results — high-dim tiers (D=32, K=64)</h2>")
    for _key, title, desc, run_dir in _HIGHDIM_TIERS:
        parts.append(f"<h3>{title}</h3>")
        parts.append(f"<p class='legend'>{desc}</p>")
        parts.append(_table(_rows(run_dir)))
    parts.append(_HIGHDIM_FINDINGS)

    parts.append(_NOISE_FINDINGS)
    parts.append(
        _figure(
            _ROOT / "outputs/noise_pareto.png",
            "MSE vs #active concepts at three within-concept-noise levels (σ=0.05/0.10/0.20); "
            "point label = recovery (matched fraction). TopK-SAE holds the upper-left "
            "(few concepts + high recovery) corner throughout; VAEE's recovery collapses as σ grows.",
        )
    )

    parts.append(_EMB_FINDINGS)
    parts.append(
        _figure(
            _ROOT / "outputs/emb_pareto.png",
            "Embedding-size Pareto (MSE vs #active concepts, σ=0.05): VAEE and VAEE-SE "
            "trajectories over E=4→128 vs the TopK / L1 baselines; point label = recovery "
            "(matched). Larger E moves both VAEE variants down-left, but on this "
            "linear-sparse tier TopK still holds the high-recovery corner.",
        )
    )

    parts.append(_INTERVENTION_FINDINGS)

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

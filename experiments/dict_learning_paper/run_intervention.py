"""Run intervention metrics (PLAN.md 6 & 7) on a finished synthetic sweep dir.

Reloads every run's checkpoint, replays the held-out validation set through the
model with single-concept ablations, and reports:
  * mean_consistency  — metric 6 (ablation-direction agreement across inputs)
  * causal_matched_fraction / causal_mean_cosine_sim — metric 7 (Hungarian-matched
    intervention directions vs the ground-truth atoms)
  * mean_dominance    — top-atom / second-atom cosine of the intervention direction

Synthetic data is unscaled, so the atoms + val data stored in ground_truth.json
are already in the model's input space.

Usage:
    python experiments/dict_learning_paper/run_intervention.py \
        experiments/dict_learning_paper/outputs/sweep_synthetic_highdim_hard_embsweep/20260616-172724
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from lcblm.eval.intervention import evaluate_intervention
from lcblm.sae_utils import TopK
from lcblm.sae_utils.model import SparseAE
from lcblm.vaee.models import VAEE, VAEESharedEncoder

_SUPPORTED = ("vaee", "vaee_shared_encoder", "topk_sae", "sae_concept")


def _build_and_load(run_dir: Path, input_dim: int) -> tuple[str, nn.Module] | None:
    cfg = json.loads((run_dir / "config.json").read_text())
    model_type = cfg["model_type"]
    rc = cfg["run_config"]
    state = torch.load(run_dir / "checkpoint.pt", map_location="cpu")

    if model_type == "vaee":
        model: nn.Module = VAEE(
            input_dim=input_dim,
            num_embeddings=rc["num_embeddings"],
            embedding_size=rc["embedding_size"],
            gumbel_temp=rc.get("gumbel_temp", 0.5),
            sigma_0=rc.get("sigma_0", 0.1),
            sim_metric=rc.get("sim_metric", "cosine"),
            topology=rc.get("topology", "stacked"),
            encoder_type=rc.get("encoder_type", "shallow"),
            hidden_dim=rc.get("hidden_dim", 256),
        )
    elif model_type == "vaee_shared_encoder":
        model = VAEESharedEncoder(
            input_dim=input_dim,
            num_embeddings=rc["num_embeddings"],
            embedding_size=rc["embedding_size"],
            gumbel_temp=rc.get("gumbel_temp", 0.5),
            sigma_0=rc.get("sigma_0", 0.1),
            sim_metric=rc.get("sim_metric", "cosine"),
            topology=rc.get("topology", "stacked"),
            encoder_type=rc.get("encoder_type", "shallow"),
            hidden_dim=rc.get("hidden_dim", 256),
            gate_mean_only=rc.get("gate_mean_only", False),
        )
    elif model_type in ("topk_sae", "sae_concept"):
        # Infer latent_dim from the checkpoint encoder weight (latent, input).
        latent_dim = state["_encoder.weight"].shape[0]
        activation = TopK(rc["k"]) if model_type == "topk_sae" else nn.ReLU()
        model = SparseAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation,
            tied_weights=False,
            use_tied_bias=True,
        )
        model.init_tied_bias(torch.zeros(input_dim))
    else:
        return None

    model.load_state_dict(state)
    model.eval()
    return model_type, model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir", help="Timestamped sweep output dir")
    p.add_argument(
        "--threshold", type=float, default=0.7, help="recovery cos threshold"
    )
    p.add_argument("--limit", type=int, default=0, help="cap val samples (0 = all)")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    gt = json.loads((sweep_dir / "ground_truth.json").read_text())
    atoms = torch.tensor(gt["atoms_2d"], dtype=torch.float32)
    x = torch.tensor(gt["val_data_2d"], dtype=torch.float32)
    if args.limit:
        x = x[: args.limit]
    input_dim = atoms.shape[1]

    rows = []
    for run_dir in sorted(sweep_dir.glob("*/")):
        if not (run_dir / "config.json").exists():
            continue
        built = _build_and_load(run_dir, input_dim)
        if built is None:
            continue
        model_type, model = built
        m = evaluate_intervention(model, model_type, x, atoms, threshold=args.threshold)
        m["run"] = run_dir.name
        rows.append(m)
        print(
            f"{run_dir.name:42s} consist={m['mean_consistency']:.3f}  "
            f"causal_matched={m['causal_matched_fraction']:.3f}  "
            f"causal_cos={m['causal_mean_cosine_sim']:.3f}  "
            f"dominance={m['mean_dominance']:.2f}  (K={m['n_concepts']})"
        )

    out = sweep_dir / "intervention.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

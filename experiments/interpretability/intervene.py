r"""Concept intervention CLI — autoregressive text generation with concept steering.

This module implements forward-hook-based concept interventions on a live
Mistral-7B model. At each generation step:
  1. A hook on the embedding layer extracts the input token embedding.
  2. The concept model encodes it to concept activations.
  3. The user-specified interventions (zero / clamp / scale) are applied.
  4. The modified activations are decoded back to an embedding.
  5. The hook replaces the original embedding with the modified one.
  6. Generation continues normally.

This depends on Mistral-7B being loaded online, so it is intentionally kept
separate from the offline analysis pipeline (train.py, build_cd.py).

Usage
-----
    interp-intervene --run-dir outputs/sst2_20260429/ --model vaee \\
        --prompt "The movie was" --zero 5,12,30 --scale 7:2.0

TODO: implement the full intervention pipeline once the offline analysis is done.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="interp-intervene",
        description="Autoregressive generation with concept interventions (not yet implemented).",  # noqa: E501
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--model",
        required=True,
        choices=["vaee", "topk_sae", "sae_concept", "sae_param"],
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--zero",
        default="",
        help="Comma-separated concept indices to zero out.",
    )
    parser.add_argument(
        "--scale",
        default="",
        help="Comma-separated idx:factor pairs to scale.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.parse_args()

    msg = (
        "intervene.py is not yet implemented. "
        "Complete the offline analysis pipeline (train.py → build_cd.py) first."
    )
    raise NotImplementedError(
        msg,
    )


if __name__ == "__main__":
    main()

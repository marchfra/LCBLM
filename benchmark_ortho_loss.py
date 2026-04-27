# ruff: noqa: N803, N806
"""Benchmark compute_decoder_ortho_loss vs compute_decoder_ortho_loss_optimized.

Measures wall time for forward + backward (matching real training conditions) on
a toy configuration and on a real-world configuration matching the Mistral-7B
SST-2 experiment (K=128, E=128, D=4096).

Usage
-----
    uv run python benchmark_ortho_loss.py
"""

from __future__ import annotations

import math
import statistics
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lcblm.utils import get_device
from lcblm.vaee.models import (
    _compute_decoder_ortho_loss_slow,
    compute_decoder_ortho_loss,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

CONFIGS: list[dict] = [
    {
        "label": "toy   (K=8,   E=8,   D=64  )",
        "K": 8,
        "E": 8,
        "D": 64,
        "n_warmup": 20,
        "n_iters": 200,
    },
    {
        "label": "real  (K=128, E=128, D=4096)",
        "K": 128,
        "E": 128,
        "D": 4096,
        "n_warmup": 5,
        "n_iters": 20,
    },
]

FN_PAIRS: list[tuple[str, Callable[[Tensor, int, int], Tensor]]] = [
    ("reference  (O(K²) loop)", _compute_decoder_ortho_loss_slow),
    ("optimized  (vectorized) ", compute_decoder_ortho_loss),
]

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

CORRECTNESS_CONFIGS = [
    {"label": "tiny  (K=4,   E=4,   D=16 )", "K": 4, "E": 4, "D": 16},
    {"label": "small (K=16,  E=8,   D=128)", "K": 16, "E": 8, "D": 128},
    {"label": "real  (K=128, E=128, D=4096)", "K": 128, "E": 128, "D": 4096},
]
N_SEEDS = 5
ATOL = 1e-3  # float32 accumulation order differs between loop and batched matmul


def check_correctness(device: torch.device) -> None:
    print("\n  Correctness check (reference vs optimized, forward value only)")
    print(f"  {'─' * 60}")
    all_ok = True
    for cfg in CORRECTNESS_CONFIGS:
        K, E, D = cfg["K"], cfg["E"], cfg["D"]
        worst_err = 0.0
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            w = torch.randn(D, K * E, device=device)  # ty:ignore[invalid-argument-type, unsupported-operator]
            ref = _compute_decoder_ortho_loss_slow(w, K, E).item()  # ty:ignore[invalid-argument-type]
            opt = compute_decoder_ortho_loss(w, K, E).item()  # ty:ignore[invalid-argument-type]
            worst_err = max(worst_err, abs(ref - opt) / (abs(ref) + 1e-8))
        ok = worst_err < ATOL
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(
            f"  [{status}]  {cfg['label']}  max relative error = {worst_err:.2e}"
            f"  (atol={ATOL})",
        )
    if not all_ok:
        msg = "Correctness check failed — do not swap in the optimized version."
        raise RuntimeError(msg)
    print("  All checks passed.\n")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_fn(  # noqa: PLR0913
    fn: Callable[[Tensor, int, int], Tensor],
    weight_template: Tensor,
    K: int,
    E: int,
    n_warmup: int,
    n_iters: int,
    *,
    use_cuda: bool,
) -> list[float]:
    """Warm up then time fn; return per-iteration wall times in milliseconds.

    On CUDA, torch.cuda.synchronize() is called before stopping the clock so
    that the measured time includes GPU kernel completion, not just launch.
    """
    # Warm-up (un-timed)
    for _ in range(n_warmup):
        w = weight_template.detach().requires_grad_(True)  # noqa: FBT003
        fn(w, K, E).backward()
    if use_cuda:
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iters):
        # Fresh leaf each iteration — mirrors optimizer.zero_grad() in training
        w = weight_template.detach().requires_grad_(True)  # noqa: FBT003
        t0 = time.perf_counter()
        fn(w, K, E).backward()
        if use_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1_000)  # ms

    return times


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(times: list[float]) -> str:
    mean = statistics.mean(times)
    if len(times) > 1:
        ci = 1.96 * statistics.stdev(times) / math.sqrt(len(times))
        return f"{mean:8.2f} ms  ±{ci:6.2f} ms  (n={len(times)})"
    return f"{mean:8.2f} ms  (n=1)"


def run_benchmark(cfg: dict, device: torch.device) -> None:
    K, E, D = cfg["K"], cfg["E"], cfg["D"]
    use_cuda = device.type == "cuda"

    print(f"\n  {cfg['label']}")
    print(f"  {'─' * 60}")

    # Single weight on device; each timed iteration detaches + re-attaches
    # requires_grad, matching the role of an nn.Parameter at the start of a step.
    weight_template = torch.randn(D, K * E, device=device)

    ref_times: list[float] | None = None
    for fn_name, fn in FN_PAIRS:
        times = _time_fn(
            fn,
            weight_template,
            K,
            E,
            cfg["n_warmup"],
            cfg["n_iters"],
            use_cuda=use_cuda,
        )
        line = f"  {fn_name}  {_fmt(times)}"
        if ref_times is not None:
            speedup = statistics.mean(ref_times) / statistics.mean(times)
            line += f"  →  {speedup:.1f}x faster"
        print(line)
        if ref_times is None:
            ref_times = times


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    device = get_device()

    print("=" * 70)
    print("  ortho loss benchmark — forward + backward (mirrors training step)")
    print(f"  device: {device}")
    print("=" * 70)

    check_correctness(device)

    print("  Timing benchmark")
    print(f"  {'─' * 60}")
    for cfg in CONFIGS:
        run_benchmark(cfg, device)

    print("=" * 70)


if __name__ == "__main__":
    main()

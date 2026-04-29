---
name: compute_decoder_ortho_loss bottleneck
description: O(K²) Python loop in ortho loss causing 7x training slowdown; vectorized fix derived and verified
type: project
---

Location: `src/lcblm/vaee/models.py::compute_decoder_ortho_loss()`

**Confirmed bottleneck:** At K=128, D=4096, E=128, the double Python loop over K*(K-1)/2=8,128 pairs:
- Launches ≥16,256 CUDA kernels (matmul + norm per pair)
- Builds an autograd graph with ~57,000 nodes
- Redundantly recomputes each block's norm O(K) times (not cached)

**Root cause of 7x slowdown:** Kernel launch overhead and Python GIL serialization dominate; the FLOP ratio (old/new) is only 1.9x.

**Vectorized fix (verified numerically across 5 test cases):**

Mathematical identity used:
- `||Wi_n^T Wj_n||_F^2 = <Wi_n Wi_n^T, Wj_n Wj_n^T>_F`  (D×D gram Frobenius inner product)
- `sum_{i<j} = (||VV^T||_F^2 - sum_i ||Wi_n Wi_n^T||_F^2) / 2`
- `||Wi_n Wi_n^T||_F^2 == ||Wi_n^T Wi_n||_F^2`  (singular value identity; verified numerically)
- Where V = normalized blocks concatenated: shape (D, K*E)

Key operations:
1. `blocks = weight.reshape(D, K, E).permute(1, 0, 2)`  → (K, D, E)
2. `blocks_n = blocks / blocks.norm(dim=(1,2), keepdim=True).clamp(min=1e-8)`  → normalized
3. `V = blocks_n.permute(1,0,2).reshape(D, K*E)`
4. `G = V @ V.T`  → (D, D), FLOPs: 2·D²·K·E = 549 GFLOPs at target size
5. `total_sq = (G*G).sum()`
6. `Gi = torch.bmm(blocks_n.permute(0,2,1), blocks_n)`  → (K, E, E)
7. `diag_sq = (Gi*Gi).sum()`
8. `return (total_sq - diag_sq) / 2.0`

**IMPORTANT: E×E gram approach is WRONG.** `<A^T A, B^T B>_F ≠ ||A^T B||_F^2` in general. Must use D×D gram.

FLOPs at K=128, D=4096, E=128:
- Old: 1,091 GFLOPs (but dominated by Python/launch overhead)
- New: 567 GFLOPs (2 GPU kernel calls)

Memory (new):
- V (D, K*E): 268 MB
- G (D, D): 67 MB
- blocks_n (K, D, E): 268 MB
- Gi (K, E, E): 8 MB

**Why:** Profiled via wall-clock measurement (65h vs 9h). Python loop over 8,128 pairs blocks GPU pipeline.
**How to apply:** Any future K-indexed penalty should use gram-based vectorization, not Python loops.

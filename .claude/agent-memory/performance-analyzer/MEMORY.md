# Performance Analyzer Memory Index

- [Project context](project_context.md) — LCBLM research codebase, frozen Mistral-7B, VAEE/SAE training on SST-2 embeddings
- [ortho_loss bottleneck](bottleneck_ortho_loss.md) — compute_decoder_ortho_loss: O(K²) loop causing 7x training slowdown; vectorized fix derived and verified

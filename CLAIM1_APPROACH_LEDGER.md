# Claim 1 approach ledger — exactly 10 routes

Target claim: one unified Regression Language Model predicts memory, latency,
and trained-neural-network accuracy from code/ONNX across multiple languages.
The official stale verdict is `toy` because the first accuracy run covered only
64 NASBench101 rows. This repair executes exactly 10 routes—no more and no
fewer. Checks within a route are subchecks, not additional approaches.

| # | Route | Scale / decisive output | State |
|---:|---|---|---|
| 1 | Primary-source scope and protocol audit | ICML paper, author repository, model/dataset cards; resolves “accuracy” to held-out neural-network `val_accuracy` from ONNX | complete |
| 2 | Unified-checkpoint identity audit | `RLM-GemmaS-Code-v0` and `RegressLM-gemma-s-RLM-table3` have identical 725,864,700-byte weights and SHA-256 `7e9df429…` | complete |
| 3 | APPS memory reproduction | 512 programs, eight draws per row, released model/recipe; Spearman `.9268` | complete |
| 4 | KernelBook latency reproduction | 512 kernels, eight draws per row, released model/recipe; Spearman `.5353` | complete |
| 5 | CodeNet multi-language memory reproduction | 17 languages × 200 rows × eight draws; mean per-language Spearman `.5299` | complete |
| 6 | NASBench101 ONNX accuracy reproduction | 512 released rows × eight draws; Spearman `.4066`, zero decode failures | complete |
| 7 | ENAS ONNX accuracy reproduction | 512 released rows × eight draws, author-card 4,096-token protocol | running |
| 8 | NASNet ONNX accuracy reproduction | 512 released rows × eight draws, author-card 4,096-token protocol | running |
| 9 | Accuracy uncertainty and permutation route | Per-space bootstrap intervals, pooled/mean correlations, and label-permutation null | pending on routes 6–8 |
| 10 | ONNX input-shuffle falsification route | Re-pair predictions and targets after deterministic input permutation; unified signal must collapse relative to paired results | pending on routes 6–8 |

Fail-closed invariant: the machine-readable validation output must report
`approaches_executed == 10`, contain route numbers 1 through 10 exactly once,
and reject any route numbered 11 or higher.

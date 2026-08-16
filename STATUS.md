# RegressLM reproduction status

Last audit: 2026-08-16

## Publication state

The repository is the paper-first reproduction surface for
*Regression Language Models for Code*. Its historical challenge result is
6/10 at judged Space revision
19231479d69a31c8e01832c24e146c37eca9a5ba. Candidate revision
f791e670006b5d428a2e11593eb29fb568bb4b45 was published to
[DineshAI/utTapVWtc7](https://huggingface.co/spaces/DineshAI/utTapVWtc7) for
live evaluation. No later score is asserted here.

The repository’s current audit has five explicit contracts:

| Contract | Status | Meaning |
| --- | --- | --- |
| C1 | VERIFIED_SCOPED | One released checkpoint produces positive memory, latency, and ONNX validation-accuracy ranking evidence |
| C2 | VERIFIED_SCOPED | APPS and KBSS Table 3 evaluation pathway is reproduced at n=512 |
| C3 | VERIFIED_SCOPED | The challenge-era 17-language CodeNet subset exceeds mean Spearman 0.5 |
| C4 | BLOCKED_LOW | Exact five-space Kendall comparison cannot be run without target-specific artifacts |
| C5 | BLOCKED_LOW | Exact head ablation and 300M/600M scaling comparisons cannot be run without artifacts |

## Verified evidence

- C1 validation has exactly 10 routes and includes 512-row NASBench101,
  ENAS, and NASNet evaluations with eight draws per row.
- C1 accuracy correlations are 0.406599, 0.249461, and 0.206738; the
  three-space mean is 0.287599 with permutation p=0.000500.
- C2 retains APPS Spearman 0.926807 and KBSS Spearman 0.535279, with
  independent statistical checks.
- C3 retains 17 languages × 200 rows, with primary mean 0.529850 and an
  independent CPU mean 0.523403. The primary bundle includes bootstrap and
  permutation controls.
- The current arXiv version discusses 24 CodeNet languages. C3 is deliberately
  labeled as a 17-language scoped result.

## Blocked evidence

C4’s exact Table 4 contract requires target-specific RLM checkpoints for the
five NAS spaces, exact few-shot and evaluation row manifests, and the training
configuration. A unified-base diagnostic and a 16-row DARTS calibration are
retained as diagnostics only.

C5’s exact Table 5 and Table 6 contracts require the three head variants, the
300M and 600M RLM checkpoints or complete training recipes, exact splits, and
seeds. Related normalization code and base-model parameter metadata are
provenance, not substitutes for the missing experiments.

The evidence-level explanation is in
[CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md); machine-readable status is in
[claims.json](claims.json).

## Resume command

~~~bash
uv run --locked python repro/src/run_campaign.py
~~~

The model and datasets are external artifacts. Pinning transformers 4.53.2,
the upstream commit, model revisions, dataset revisions, and the exact
sampling protocol is required; see [ENVIRONMENT.md](ENVIRONMENT.md) and
[SOURCE_AUDIT.md](SOURCE_AUDIT.md).

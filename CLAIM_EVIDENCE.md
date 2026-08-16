# Claim-to-evidence map

This file explains how every claim in this repository is produced, what
evidence is retained, and where the evidence stops. The status vocabulary is
deliberately conservative:

- VERIFIED_SCOPED means the exact contract written here is supported by
  retained evidence, with its population, checkpoint, and protocol stated.
- BLOCKED_LOW means the audit found a material artifact gap and the exact
  claim cannot be evaluated without inventing an experiment identity.
- A diagnostic is never promoted to the paper claim when its model, rows,
  scale, or target differs from the contract.

## Shared experiment identity

All positive evaluations use the released Table 3 checkpoint
akhauriyash/RegressLM-gemma-s-RLM-table3 at revision
5e5002672f870399ce012896332363e271582509. The checkpoint contains
181,458,944 parameters and its common weights hash is
7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c.

For stochastic evaluations, the protocol is:

1. Construct the dataset-specific input prefix.
2. Generate eight predictions with do_sample=true, top_p=0.95,
   temperature=1.0, and max_new_tokens=9.
3. Decode the first numeric value and take the median of the eight draws.
4. Compute the requested Spearman or Kendall statistic.
5. Preserve row-level results, raw draws, environment metadata, and hashes
   whenever the artifact is part of the headline result.

The model export requires transformers 4.53.2. The external upstream checkout
is pinned to google-deepmind/regress-lm commit 6c23ccb. Full pins and input
prefixes are in [ENVIRONMENT.md](ENVIRONMENT.md) and
[SOURCE_AUDIT.md](SOURCE_AUDIT.md).

## C1 — one checkpoint, three modalities

### Contract

One released RLM checkpoint should produce useful ranking signal for memory,
latency, and validation accuracy. The validation-accuracy modality means
trained neural-network validation accuracy from ONNX architecture records; it
does not mean functional correctness of ordinary code.

### Production path

1. APPS and KBSS use metric-valued targets from the Code-Regression source.
   The evaluator applies the required space prefix, samples eight outputs per
   row, takes the median, and computes Spearman correlation.
2. GraphArch-Regression supplies ONNX architecture records and the val_accuracy
   target for NASBench101, ENAS, and NASNet.
3. run_campaign.py coordinates the fixed campaign. The accuracy runner and
   independent checkers recompute the row medians and statistics.
4. claim1_validation.json records the exactly-10-route gate. The routes are
   fixed in CLAIM1_APPROACH_LEDGER.md; adding another route would change the
   contract rather than strengthen this result.

### Retained result

The paper-scale accuracy routes retain 512 rows per space and eight draws per
row:

| Space | Spearman |
| --- | ---: |
| NASBench101 | 0.4065993611 |
| ENAS | 0.2494611775 |
| NASNet | 0.2067375234 |
| Three-space mean | 0.2875993540 |

The permutation control has p=0.0004997501 and the input-shuffle control has
mean correlation -0.013240. The source audit also verifies that the unified
model alias and the Table 3 alias have identical critical files and weights.

### Scope limitation

The checkpoint is smaller than the paper’s rounded 300M label: the local
checkpoint has 181.5M parameters. The result is therefore a
VERIFIED_SCOPED reproduction of the released checkpoint’s three modalities,
not a claim that the exact paper training run or rounded model-size label has
been reconstructed.

Primary evidence:

- outputs/claim1_validation.json
- outputs/claim1_source_audit.json
- outputs/claim1_accuracy/
- outputs/colab/table3_results.json
- repro/src/verify_cumulative.py

## C2 — APPS and KBSS correlation pathway

### Contract

Evaluate the released checkpoint on the retained APPS and KBSS Table 3
pathway at n=512 per dataset and reproduce positive Spearman ranking signal.
The target is a numeric performance metric, not a binary code-correctness
label.

### Production path

- APPS and KBSS inputs are prefixed with the dataset name and newline.
- The target is the metric value: memory bytes for APPS and latency in
  milliseconds for KBSS.
- Eight stochastic decodes are reduced to a median per row.
- scipy.stats.spearmanr and the independent verifier calculate the statistic.
- The result is compared to the challenge-era references, while row-level
  and model revisions remain visible.

### Retained result

| Dataset | Rows | Spearman |
| --- | ---: | ---: |
| APPS | 512 | 0.9268067718 |
| KBSS | 512 | 0.5352789638 |

These values establish the scoped evaluation path. They do not imply that
every Table 3 number, dataset split, or current v3 paper claim is identical to
the released card.

Primary evidence:

- outputs/colab/table3_results.json
- outputs/phaseA/independent_verification.json
- repro/src/run_eval.py
- repro/src/verify_independent.py

## C3 — 17-language CodeNet result

### Contract

On the challenge-era 17-language CodeNet subset, the mean of per-language
Spearman values is greater than 0.5. The evaluation uses 200 rows per
language, eight stochastic draws per row, median aggregation, and
language-stratified uncertainty and permutation controls.

### Production path

1. The evaluator selects the fixed top-17 language set from the Code-Regression
   source and applies the CDSS language prefix.
2. It evaluates 200 rows per language and retains raw draws.
3. verify_codenet.py independently recomputes per-language medians, the
   unweighted mean, bootstrap interval, permutation p-value, and shuffled
   target control.
4. A second independent run on CPU is retained as corroboration, not merged
   into the primary estimate.

### Retained result

- Primary Colab bundle: mean Spearman 0.5298501743.
- Independent CPU run: mean Spearman 0.5234034026.
- Primary stratified bootstrap 95% interval:
  [0.5025569766, 0.5542456354].
- Primary one-sided permutation p=0.0004997501.
- The primary bundle contains 3,464 raw rows and 27,712 raw draws.

### Version limitation

The current arXiv v3 abstract describes 24 CodeNet languages, whereas the
challenge-era contract and this retained run cover 17. C3 is therefore
VERIFIED_SCOPED for 17 languages only. It is not evidence for the full current
24-language statement.

Primary evidence:

- outputs/colab/evidence_bundle_verification.json
- outputs/codenet/full_gpu_n200_verification.json
- outputs/colab/regresslm_evidence_bundle.zip
- outputs/codenet/full_gpu_n200.csv
- repro/src/verify_codenet.py
- repro/src/verify_evidence_bundle.py

## C4 — exact Table 4 Kendall comparison

### Contract

Reproduce the paper’s five-space target-specific Kendall tau-b comparison and
its reported average 0.461, with the exact few-shot protocol, target
checkpoints, row identities, and comparator thresholds.

### Four audit routes

1. Source and protocol reconstruction verifies the displayed arithmetic and
   the five-space interpretation, but the ENAS and NASNet target-specific
   checkpoints and all row manifests are absent.
2. An independent Kendall recomputation on retained unified-base ENAS and
   NASNet predictions agrees exactly, but those predictions come from the
   wrong checkpoint for the exact contract.
3. A released DARTS checkpoint executes end to end on 16 deterministic rows
   with eight draws, producing tau-b 0.55. It is a bounded calibration, not
   the full target-domain comparison.
4. A dedicated falsification search requires all five target-specific
   prediction sets and exact row identities. No valid counterexample can be
   constructed because the required artifacts are absent.

### Verdict

BLOCKED_LOW. The independent checker reports PASS because the blocked record,
table arithmetic, metric implementations, and negative controls are
internally consistent. PASS here means the audit is coherent; it does not mean
that Table 4 was reproduced.

Evidence directory:

- .openresearch/artifacts/claim4_released_audit/four_routes.json
- .openresearch/artifacts/claim4_released_audit/independent_checker_output.json
- .openresearch/artifacts/claim4_released_audit/negative_control_output.json
- .openresearch/artifacts/claim4_released_audit/source_audit.md
- repro/src/claim4_exact_gate.py

Unblockers are the pinned ENAS and NASNet target checkpoints, all five
few-shot/evaluation row manifests, and the exact training configuration.

## C5 — exact head ablation and model scaling

### Contract

Reproduce the paper’s Table 5 decoder-head comparison
(0.800 versus 0.717 versus 0.478) and Table 6 scaling comparison
(600M 0.782 versus 300M 0.744), using the exact implementations, data
splits, seeds, checkpoints, and evaluation rows.

### Four audit routes

1. Source and official-artifact inspection reconstructs the table values and
   detects a synthetic control path, but finds no exact Table 5 artifacts or
   public 600M RLM.
2. The cited normalized-head implementation is pinned and its normalization
   behavior is verified. It is related provenance, not the paper’s exact
   head, split, checkpoint, or training run.
3. Base T5Gemma parameter metadata and the reported 0.038 performance
   difference are independently recomputed. Parameter counts cannot establish
   the missing performance result.
4. A dedicated falsification route requires raw predictions from all exact
   formulations and model sizes. The required prediction pairs are absent, so
   no valid counterexample is constructed.

### Verdict

BLOCKED_LOW. The independent checker reports PASS for the internal consistency
of the blocked record, while its interpretation explicitly rejects treating
the related source or base-model metadata as claim evidence.

Evidence directory:

- .openresearch/artifacts/claim5_ablation_scaling/four_routes.json
- .openresearch/artifacts/claim5_ablation_scaling/independent_checker_output.json
- .openresearch/artifacts/claim5_ablation_scaling/negative_control_output.json
- .openresearch/artifacts/claim5_ablation_scaling/source_audit.md
- repro/src/claim5_exact_gate.py

Unblockers are exact checkpoints or complete training recipes for the three
Table 5 heads, exact Table 5 rows and seeds, and pinned 300M/600M RLM
checkpoints or complete same-settings recipes with exact Table 6 rows.

## Reproducibility rule

The evidence files are useful because they preserve both positive results and
the reasons an exact claim could not be evaluated. Any future run should add a
new, explicitly scoped contract if it changes the model, paper version,
language population, row set, or metric. It should not silently replace the
contracts above.

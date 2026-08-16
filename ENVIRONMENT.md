# Reproduction environment and protocol

This file records the execution identity needed to interpret the retained
results. It intentionally separates local CPU corroboration from the Colab
T4 evidence bundle.

## Software pins

- Python: 3.12
- Package manager: uv
- Lockfile: uv.lock
- transformers: 4.53.2
- Upstream source: google-deepmind/regress-lm commit 6c23ccb
- Primary run command: uv run --locked python repro/src/run_campaign.py
- Random seed used by the retained evidence: 42

The checkpoint’s config records transformers 4.53.2. Do not replace it with
transformers 5.x: the generation implementation changes the encoder-to-
decoder path for this custom T5Gemma model and can produce input-insensitive
predictions.

## Model and data

- Checkpoint:
  akhauriyash/RegressLM-gemma-s-RLM-table3
- Checkpoint revision:
  5e5002672f870399ce012896332363e271582509
- Model parameter count: 181,458,944
- Common weights SHA-256:
  7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c
- Unified alias audit revision:
  akhauriyash/RLM-GemmaS-Code-v0 at
  0c927733af21f156d61743c4a40d03d13e65c16b
- Code-Regression: public dataset; the full parquet is approximately 5.6 GB
- GraphArch-Regression revision:
  c557392740094b539bbdb527d03e3a78e5b34a38
- GraphArch parquet size: 4,116,945,368 bytes
- GraphArch parquet SHA-256:
  2a5992248d27a060c031d7a9485207310a77f58bd2c289cbe37d53d0fd894ce0

The model weights, large parquet files, and upstream checkout are external
artifacts. They are excluded from the normal clone and must be downloaded at
the pinned revisions.

## Shared decoding protocol

For every retained stochastic result:

1. Use the dataset-specific input prefix.
2. Generate with do_sample=true, top_p=0.95, temperature=1.0,
   min_new_tokens=max_new_tokens=9, and use_cache=true.
3. Decode the first numeric output.
4. Take the median of eight draws for each row.
5. Compute the correlation on the row-level medians.

Input prefixes:

- APPS and KBSS: SPACE, newline, INPUT
- CDSS: the CDSS marker, language marker, newline, and INPUT
- GraphArch: the released ONNX architecture text and its val_accuracy target

## Evidence runs

### C1 accuracy

- Spaces: NASBench101, ENAS, NASNet
- Rows per space: 512
- Draws per row: 8
- Retained row count: 1,536
- Retained draw count: 12,288
- Device: Colab Tesla T4 for the primary bundle
- Precision: bfloat16 in the primary bundle
- Primary mean Spearman: 0.2875993540
- Input-shuffle mean: -0.013240
- Permutation p-value: 0.0004997501

### C2 metric pathway

- APPS rows: 512
- KBSS rows: 512
- Draws per row: 8
- APPS Spearman: 0.9268067718
- KBSS Spearman: 0.5352789638
- Independent Phase A check: 40 APPS rows, Spearman 0.9374237733,
  Pearson 0.9202369676, shuffled control -0.205

### C3 CodeNet

- Languages: the fixed challenge-era top 17
- Rows per language: 200
- Draws per row: 8
- Primary device: Colab Tesla T4
- Primary environment: Python 3.12.13, Linux x86_64, torch 2.11.0+cu128,
  CUDA 12.8
- Primary raw rows: 3,464
- Primary raw draws: 27,712
- Primary mean Spearman: 0.5298501743
- Independent CPU mean Spearman: 0.5234034026
- Primary stratified bootstrap 95% interval:
  [0.5025569766, 0.5542456354]
- Primary permutation p-value: 0.0004997501
- Primary shuffled-target control: 0.0351678566

## Blocked-claim audits

C4’s DARTS calibration used 16 released rows and 128 raw draws; its tau-b
value of 0.55 is a bounded diagnostic. It is not the five-space Table 4
result.

C5’s audit uses pinned public metadata and a related normalization source. It
does not train or infer the exact three head variants or the 300M/600M RLM
pair. The blocked verdict is therefore an artifact-availability conclusion,
not a performance estimate.

## Verification commands

Lightweight publication checks:

~~~bash
python3 verify_final.py
python3 -m json.tool claims.json >/dev/null
python3 -m json.tool EVIDENCE_MANIFEST.json >/dev/null
git diff --check
~~~

The verifier checks the claim gates, evidence JSON, branch inventory, origin
identity, commit attribution, and manifest hashes. It does not download model
weights or rerun the expensive inference campaign.

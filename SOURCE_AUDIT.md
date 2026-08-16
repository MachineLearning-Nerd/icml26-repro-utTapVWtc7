# Source and provenance audit

This audit separates the current paper source, the challenge-era contract,
the official implementation, and the released evaluation artifacts. A paper
version or public artifact is not treated as interchangeable merely because it
has the same title.

## Paper sources

| Source | Retrieval artifact | SHA-256 | Use |
| --- | --- | --- | --- |
| arXiv v1 PDF | paper_2509.26476v1.pdf | fb8f644ea37b3f3ff9201ea76613bf5bd35a4ddb7445b1a8e15fb59a8e5cd43d | Challenge-era source comparison |
| arXiv v1 source | source/arxiv/2509.26476v1.tar | f58102fb290dc9a95542089911b9559c01385be821c5c3df59a037195fa00a6f | TeX/source inspection |
| arXiv v3 PDF | paper_2509.26476v3.pdf | 932f9b77f16a99d6ceafeedf69c97affcdf956e3aa93357031534c6ddca8f9eb | Current ICML 2026 paper record |
| arXiv v3 source | source/arxiv/2509.26476v3.tar | 6b5579b681650a527a578b11a3a0c5e039f07eafaf01c987f04bb196a9422677 | Current TeX/source inspection |

Canonical web records:

- [Current arXiv record](https://arxiv.org/abs/2509.26476)
- [Challenge OpenReview record](https://openreview.net/forum?id=utTapVWtc7)
- [ICML 2026 reproduction challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)

The current arXiv record is version 3, revised 2026-06-16, and identifies the
paper as an ICML 2026 publication. The v1 and v2 abstract wording reports the
17-language CodeNet result; the v3 abstract reports 24 languages. The v1/v2
Table 3 already describes a 24-language filtered evaluation table. The
challenge contract retained by this repository is the 17-language subset,
so the version boundary is explicit rather than silently normalized.

## Paper identity

- Title: Regression Language Models for Code
- Authors: Yash Akhauri; Xingyou Song; Arissa Wongpanich; Bryan Lewandowski;
  Mohamed S. Abdelfattah
- arXiv identifier: 2509.26476
- Current version: v3
- Conference context: International Conference on Machine Learning (ICML 2026)
- Challenge identifier: utTapVWtc7

The paper’s central object is a regression language model that maps source code
or problem text to numeric performance metrics. The three modalities in the
reproduction are memory, latency, and trained-neural-network validation
accuracy. The latter comes from ONNX architecture records and is not program
functional correctness.

## Official implementation and model

| Artifact | Pin or revision | Role |
| --- | --- | --- |
| google-deepmind/regress-lm | commit 6c23ccb | Upstream implementation |
| akhauriyash/RegressLM-gemma-s-RLM-table3 | revision 5e5002672f870399ce012896332363e271582509 | Released Table 3 checkpoint |
| akhauriyash/RLM-GemmaS-Code-v0 | revision 0c927733af21f156d61743c4a40d03d13e65c16b | Unified model alias audit |
| akhauriyash/GraphArch-Regression | revision c557392740094b539bbdb527d03e3a78e5b34a38 | ONNX architecture/accuracy data |
| akhauriyash/Code-Regression | public dataset | APPS, KBSS, and CodeNet metric data |

The common model weights hash is
7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c. The
GraphArch parquet retained for the accuracy audit has 4,116,945,368 bytes and
SHA-256
2a5992248d27a060c031d7a9485207310a77f58bd2c289cbe37d53d0fd894ce0.

Public artifact links:

- [Official code](https://github.com/google-deepmind/regress-lm)
- [Released checkpoint](https://huggingface.co/akhauriyash/RegressLM-gemma-s-RLM-table3)
- [Unified model alias](https://huggingface.co/akhauriyash/RLM-GemmaS-Code-v0)
- [GraphArch-Regression](https://huggingface.co/datasets/akhauriyash/GraphArch-Regression)
- [Code-Regression](https://huggingface.co/datasets/akhauriyash/Code-Regression)

## Evaluation-card and source semantics

The released evaluation recipe is reproduced with:

- APPS and KBSS input prefixes of the form SPACE followed by a newline and
  the input.
- CDSS input prefix containing the literal CDSS marker and language.
- Eight sampled decodes, top_p 0.95, temperature 1.0, and nine-token output.
- First numeric token sequence decoded to a float, then median aggregation.
- Spearman correlation on the metric value.

For APPS and CDSS, the metric is memory bytes; for KBSS it is latency in
milliseconds. The GraphArch accuracy route uses val_accuracy from ONNX
architecture records. These semantics are recorded in
CLAIM1_SOURCE_AUDIT.md and the executable evaluators.

The checkpoint metadata reports transformers 4.53.2. The repository therefore
pins transformers 4.53.2 even though an upstream extras specification can
select a 5.x release. The latter generation path was observed to remove the
encoder signal for this checkpoint and is not the same experiment.

## Claim-version matrix

| Repository contract | Paper/source population | Deliberate limit |
| --- | --- | --- |
| C1 | Released Table 3 checkpoint plus GraphArch NASBench101, ENAS, and NASNet | Checkpoint is 181.5M parameters; exact paper 300M training is not claimed |
| C2 | Challenge-era APPS and KBSS evaluation path at n=512 | Does not assert every Table 3 dataset or split |
| C3 | Challenge-era 17-language CodeNet subset | Does not assert the current v3 24-language result |
| C4 | Current Table 4 wording and displayed five-space values | Blocked without target-specific models and row manifests |
| C5 | Current Table 5/6 wording and displayed values | Blocked without exact heads, sizes, splits, and checkpoints |

## Source availability rule

Only artifacts that are publicly inspectable and pinned can serve as direct
claim evidence. A related implementation, a model with a different parameter
count, a synthetic detector control, or a bounded calibration may document an
audit path but cannot replace the missing exact artifact.

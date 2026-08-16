# Regression Language Models for Code

Paper-first reproduction and claim audit for the ICML 2026 paper
*Regression Language Models for Code* (RegressLM).

## Current status

This repository is a scoped, evidence-backed reproduction. Claims 1–3 are
verified within the released-checkpoint and 17-language challenge scope.
Claims 4–5 are explicitly blocked at low confidence because the target
checkpoints, split manifests, and exact experiment implementations needed for
a faithful test are not public.

The last public challenge result recorded in the repository is 6/10 at judged
Space revision
19231479d69a31c8e01832c24e146c37eca9a5ba. The later candidate revision
f791e670006b5d428a2e11593eb29fb568bb4b45 was published for evaluation; this
repository does not claim a new judge score.

## Repository identity

- Current name: MachineLearning-Nerd/icml26-regression-language-models-code
- Previous name: MachineLearning-Nerd/icml26-repro-utTapVWtc7
- Challenge record: OpenReview utTapVWtc7
- Reproduction owner: MachineLearning-Nerd
- Reproduction command: uv run --locked python repro/src/run_campaign.py

The repository name describes the paper rather than the opaque challenge
identifier. The complete old-to-new branch map is in
[BRANCH_AUDIT.md](BRANCH_AUDIT.md).

## Paper identity and version boundary

The current paper record is:

- Title: *Regression Language Models for Code*
- Authors: Yash Akhauri, Xingyou Song, Arissa Wongpanich, Bryan Lewandowski,
  and Mohamed S. Abdelfattah
- Current paper: [arXiv:2509.26476](https://arxiv.org/abs/2509.26476)
- Challenge record: [OpenReview: utTapVWtc7](https://openreview.net/forum?id=utTapVWtc7)
- Official implementation: [google-deepmind/regress-lm](https://github.com/google-deepmind/regress-lm)

The archived challenge contract and this reproduction evaluate the 17-language
CodeNet subset used by the challenge-era evidence. The current arXiv version
describes a 24-language CodeNet result. Therefore the verified Claim 3 below
must not be presented as a reproduction of the full current 24-language
abstract claim. The version comparison and hashes are recorded in
[SOURCE_AUDIT.md](SOURCE_AUDIT.md).

## What the paper does

RegressLM treats source code and problem text as inputs to a sequence-to-
sequence model and predicts numeric performance metrics directly. A
T5Gemma-based encoder represents the input; a numeric decoder emits floating
point values. Depending on the dataset, the target is memory, execution
latency, or validation accuracy of a trained neural-network architecture
represented in ONNX. The model is intended to learn one code-to-metric
interface that transfers across languages and performance modalities.

This distinction matters: the accuracy target in the NAS experiments is
trained-network validation accuracy, not pass/fail correctness of an ordinary
program.

## Claim scorecard

| Claim | Contract used here | Status | Evidence headline |
| --- | --- | --- | --- |
| C1 | One released checkpoint predicts memory, latency, and ONNX validation accuracy | VERIFIED_SCOPED | Positive correlations on APPS/KBSS and NASBench101, ENAS, and NASNet; exactly 10 validation routes |
| C2 | Released checkpoint reaches the APPS and KBSS Table 3 correlation pathway | VERIFIED_SCOPED | APPS Spearman 0.926807 and KBSS Spearman 0.535279, n=512 each |
| C3 | Released checkpoint exceeds 0.5 mean Spearman on the 17-language CodeNet subset | VERIFIED_SCOPED | Primary mean 0.529850; independent CPU run 0.523403 |
| C4 | Exact five-space Kendall comparison in Table 4 | BLOCKED_LOW | Missing target-specific ENAS/NASNet checkpoints and all row manifests |
| C5 | Exact Table 5 head ablation and Table 6 300M/600M scaling | BLOCKED_LOW | Missing exact implementations, checkpoints, splits, and the 600M RLM |

“Verified” means that the stated scoped contract is supported by retained
raw or independently recomputable evidence. It does not silently upgrade a
subset result into a broader paper claim. The detailed production paths,
controls, and blockers are in
[CLAIM_EVIDENCE.md](CLAIM_EVIDENCE.md), while machine-readable contracts are
in [claims.json](claims.json).

## How each claim is produced

| Claim | Inputs and protocol | Code path | Retained evidence |
| --- | --- | --- | --- |
| C1 | One 181,458,944-parameter released checkpoint; 512 rows per NAS space; eight stochastic draws per row; median prediction; Spearman correlation | repro/src/run_campaign.py, repro/src/run_grapharch.py, repro/src/verify_claim1_accuracy.py | outputs/claim1_validation.json, outputs/claim1_accuracy/, outputs/claim1_source_audit.json |
| C2 | APPS and KBSS rows; required dataset prefix; eight draws; median decoded metric; SciPy Spearman statistic | repro/src/run_eval.py, repro/src/verify_independent.py | outputs/colab/table3_results.json, outputs/phaseA/independent_verification.json |
| C3 | 17 CodeNet languages, 200 rows per language, eight draws per row; language-stratified mean and bootstrap/permutation controls | repro/src/run_codenet.py, repro/src/verify_codenet.py, repro/src/verify_evidence_bundle.py | outputs/colab/evidence_bundle_verification.json, outputs/codenet/full_gpu_n200_verification.json, raw CSV/ZIP evidence |
| C4 | Four-route audit of exact Table 4 protocol, metric, target checkpoints, row manifests, and a dedicated falsification attempt | repro/src/check_claim4_audit.py, repro/src/verify_claim4_audit.py, repro/src/claim4_exact_gate.py | .openresearch/artifacts/claim4_released_audit/ |
| C5 | Four-route audit of exact Table 5/6 formulations, model sizes, training identities, and a dedicated falsification attempt | repro/src/verify_claim5_audit.py, repro/src/verify_claim5_final.py, repro/src/claim5_exact_gate.py | .openresearch/artifacts/claim5_ablation_scaling/ |

All five contracts are fail-closed. A missing artifact is recorded as missing;
it is never converted into a passing result by substituting a nearby model,
dataset, or diagnostic.

## Reproduction protocol

The lockfile targets Python 3.12. The public source, model, and dataset are
external artifacts and are intentionally not copied into this repository:

1. Clone google-deepmind/regress-lm at commit 6c23ccb into upstream/.
2. Download the released checkpoint and data revisions listed in
   [SOURCE_AUDIT.md](SOURCE_AUDIT.md).
3. Install the locked environment and run:

~~~bash
uv run --locked python repro/src/run_campaign.py
~~~

The checkpoint was exported with transformers 4.53.2. Using a newer
transformers 5.x generation path can remove the encoder signal and produce
near-constant predictions, so the pinned version is part of the experiment
identity. The full environment, seeds, sample counts, prefixes, and
hardware-specific evidence are in [ENVIRONMENT.md](ENVIRONMENT.md).

## Final branch map

The publication surface is main. Descriptive audit and release branches
preserve the useful experiment lineage:

| Branch | Purpose |
| --- | --- |
| main | Paper-first README, current scorecard, and release surface |
| audit/claim-4-blocked-verdict | Final four-route Claim 4 assessment |
| audit/claim-4-config-compatibility | Bounded accepted-config calibration |
| audit/claim-4-cpu-inference | CPU inference calibration |
| audit/claim-4-evidence | Earlier Claim 4 evidence record |
| audit/claim-4-full-parquet | Claim 4 data/provenance calibration |
| audit/claim-4-kendall-checkpoint | Kendall metric and released-checkpoint audit |
| audit/claim-5-ablation-scaling | Final Table 5/6 audit |
| audit/claim-5-evidence | Claim 5 evidence record |
| audit/claim-5-source-provenance | Cited normalized-head source audit |
| audit/claim-5-source-transport | Claim 5 source-transport repair |
| audit/codenet-dual-provenance | Independent CodeNet provenance tracks |
| baseline/frozen-6-10 | Frozen cumulative challenge baseline |
| release/evaluator-visible | Evaluator-visible cumulative release |
| release/standalone-verifiers | Standalone downloaded-release verifiers |

The exact old branch names, old tips, and rename rationale are preserved in
[BRANCH_AUDIT.md](BRANCH_AUDIT.md).

## Evidence, limitations, and provenance

- The released checkpoint is the Table 3 alias
  akhauriyash/RegressLM-gemma-s-RLM-table3, revision
  5e5002672f870399ce012896332363e271582509.
- Its common weights hash is
  7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c.
- The unified model alias akhauriyash/RLM-GemmaS-Code-v0 was checked for
  identical critical files and weights; it is not treated as a separate model.
- The GraphArch-Regression dataset is pinned to revision
  c557392740094b539bbdb527d03e3a78e5b34a38; its retained parquet hash is
  2a5992248d27a060c031d7a9485207310a77f58bd2c289cbe37d53d0fd894ce0.
- C1’s local checkpoint has 181.5M parameters, while the paper uses a rounded
  300M label. That discrepancy is disclosed rather than hidden.
- The current v3 paper’s 24-language statement and the challenge-era
  17-language contract are both preserved in the source audit.
- Raw large datasets, model weights, and the vendored upstream checkout remain
  external and are not required for a documentation-only clone.

The fail-closed publication verifier is [verify_final.py](verify_final.py).
The evidence manifest is [EVIDENCE_MANIFEST.json](EVIDENCE_MANIFEST.json).

## Citation

If this reproduction or its audit artifacts are useful, please cite the paper
and this repository. A ready-to-use citation is in
[CITATION.cff](CITATION.cff).

~~~bibtex
@inproceedings{akhauri2026regression,
  title     = {Regression Language Models for Code},
  author    = {Akhauri, Yash and Song, Xingyou and Wongpanich, Arissa
               and Lewandowski, Bryan and Abdelfattah, Mohamed S.},
  booktitle = {International Conference on Machine Learning},
  year      = {2026},
  note      = {ICML 2026; arXiv:2509.26476}
}
~~~

## Thank you

Thank you to Yash Akhauri, Xingyou Song, Arissa Wongpanich, Bryan Lewandowski,
and Mohamed S. Abdelfattah for the paper, the public regress-lm
implementation, the released checkpoints, and the GraphArch-Regression and
Code-Regression data resources. Thanks also to the Google DeepMind and Cornell
research communities and to the ICML 2026 reproduction organizers for making
the artifacts and evaluation contract available. This repository is an
independent reproduction and audit; it is not an official author release.

# Claim 5 source audit

## Retrieval

- Paper: *Regression Language Models for Code*, arXiv:2509.26476.
- Source URL: https://ar5iv.labs.arxiv.org/html/2509.26476
- Retrieval date: 2026-07-27 UTC.
- Request User-Agent: `OpenResearch-Reproduction/1.0 paper-2509.26476`.
- HTML SHA-256: `5947f4512cc86850a63409adf52af25ac1f40b15dcc797348fd8ae91a2740913`.
- Anchors: Section 6.2 (`S6.SS2`), Table 5 (`S6.T5`), and Table 6 (`S6.T6`).

## Exact statements and quantifiers

Table 5 is a three-way ablation evaluated on 512 NASBench101 validation
samples. The displayed Spearman correlations are 0.478 for a standard
regression head, 0.717 for a normalized regression head, and 0.800 for the
decoder head. The surrounding text says the training subset contains
NASBench101, SNAS, OFA ResNet, OFA ProxylessNAS, and OFA MobileNet. It describes
the standard/normalized alternatives as encoder-only four-layer models trained
with MSE and the decoder formulation as a two-encoder/two-decoder model trained
with cross-entropy.

Table 6 is a two-size comparison evaluated on 1,024 CodeNet samples. It reports
0.744 for T5Gemma s-s prefix-LM (shown as 300M) and 0.782 for T5Gemma b-b
prefix-LM (shown as 600M). The surrounding text says both use exactly the same
settings and that training uses a smaller subset of CodeNet, APPS, and
KernelBook.

Appendix C states the common optimizer family and broad settings: Adafactor,
pretraining learning rate 1e-3, fine-tuning learning rate 5e-5, clipping at 2,
10% warmup followed by cosine decay, usual decoder depth 2, hidden size 2048,
8 attention heads, median of 64 inference samples, and maximum input length
2048. These statements do not identify the exact Table 5/6 rows, training
steps, seeds, checkpoint-selection rule, complete head implementation, or
checkpoints.

## Missing identifiers

The exact training and evaluation row IDs are not stated. The smaller subset in
Table 6 is not enumerated. The official repository snapshots and author model
listing are therefore audited as public evidence, but non-discovery is recorded
only as a blocker—not as proof of nonexistence or as falsification.

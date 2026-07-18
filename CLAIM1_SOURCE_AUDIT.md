# Claim 1 source and scope audit

## What “accuracy” means

The official claim compresses the paper abstract into “accuracy of code,” which
can sound like functional correctness of ordinary source programs. The primary
paper is unambiguous: the third modality is the **accuracy and speed of trained
neural networks represented in ONNX**. Section 4.2 defines the NAS target as
validation accuracy after training an architecture under fixed hyperparameters.
Table 2 lists accuracy coverage for nine NAS search spaces; ordinary APPS and
CodeNet programs provide memory targets, not pass/fail accuracy targets.

- Paper: <https://arxiv.org/pdf/2509.26476>
- ICML OpenReview record: <https://openreview.net/forum?id=utTapVWtc7>
- Author GraphArch dataset card:
  <https://huggingface.co/datasets/akhauriyash/GraphArch-Regression>

The previous NASBench101 experiment therefore used the correct target
(`val_accuracy`), but 64 rows were too small for the official judge. The repair
uses the author-card evaluation scale and spaces: 512 rows each from
NASBench101, ENAS, and NASNet, with eight stochastic draws and a median point
prediction.

## One checkpoint, not three separately fitted models

The public unified alias `akhauriyash/RLM-GemmaS-Code-v0` declares both
`GraphArch-Regression` and `Code-Regression` as training datasets. Its
`model.safetensors` object is exactly the same 725,864,700-byte object as
`akhauriyash/RegressLM-gemma-s-RLM-table3`, the checkpoint already used for the
accepted APPS/KernelBook/CodeNet runs:

```text
SHA-256 7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c
```

Thus the cross-domain result is genuinely produced by one frozen checkpoint.
No per-space fine-tuning or regression head is introduced in this repair.

- Unified model card: <https://huggingface.co/akhauriyash/RLM-GemmaS-Code-v0>
- Table-3 alias: <https://huggingface.co/akhauriyash/RegressLM-gemma-s-RLM-table3>

## Frozen revisions and protocol

- Paper: arXiv `2509.26476`; OpenReview `utTapVWtc7`.
- Author source: `google-deepmind/regress-lm@6c23ccb`.
- GraphArch dataset revision:
  `c557392740094b539bbdb527d03e3a78e5b34a38`.
- Unified-model revision:
  `0c927733af21f156d61743c4a40d03d13e65c16b`.
- Compatible Transformers version: `4.53.2` (recorded by the checkpoint and
  recommended by its card).
- Generation: `do_sample=True`, `top_p=.95`, temperature `1`, exactly nine
  output tokens, eight draws, median aggregation.
- Input: `<space>\n\n<ONNX readable graph>`, truncated to 4,096 tokens as in
  the author dataset-card evaluator.

The paper's general Appendix-C default is 2,048 tokens, while its sequence
ablation includes 4,096 and the released GraphArch evaluation code explicitly
uses 4,096. We follow the executable released evaluation contract and disclose
that choice; it is a protocol subcheck inside routes 6–8, not an eleventh route.

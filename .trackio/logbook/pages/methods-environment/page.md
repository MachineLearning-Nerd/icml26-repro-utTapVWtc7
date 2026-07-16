# Methods & environment


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_c6115f20177b", "created_at": "2026-07-16T12:41:51+00:00", "title": "Eval recipe + the transformers-version fix"}
-->
Released checkpoint, no retraining, inference-only. Recipe = the authors' published dataset-card snippet:
- Input prefix (REQUIRED): APPS/KBSS → '{SPACE}
{input}'; CDSS → '# CDSS
# Language: {lang}
{input}'.
- generate(do_sample=True, top_p=0.95, temperature=1.0, min=max_new_tokens=9, use_cache=True), 8 samples → median → scipy.spearmanr.
- Decoder = IEEE 9-token numeric tokenizer, 13-token vocab; decode via token_ids_to_floats.

**Critical fix:** config.transformers_version=4.53.2 (the export version). The regress-lm[extras] pin 'transformers>5.0.0' installs 5.x, whose rewritten seq2seq generate() breaks this T5Gemma model — the encoder signal never reaches the decoder, so output collapses to a constant ≈0 regardless of input (ρ≈0). Pinning transformers==4.53.2 + use_cache=True restores input sensitivity: greedy then tracks targets (6036→5720, 13235→13137).

target column = metric value (memory bytes for APPS/CDSS; latency ms for KBSS), NOT accuracy (the card's 'val_accuracy' label is misleading). Spearman is rank-based.

Environment: py3.12 venv, transformers 4.53.2, torch 2.13.0+cu130 (CPU; GTX 1050 sm_61 unsupported by cu130). eval driver: repro/src/run_eval.py.

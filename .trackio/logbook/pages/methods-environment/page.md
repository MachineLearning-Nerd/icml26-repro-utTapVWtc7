# Methods & environment


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_claim1_methods_final", "created_at": "2026-07-18T23:08:23+00:00", "title": "Released-source protocol, checkpoint identity, and independent verification"}
-->
The repair uses released artifacts without retraining. Accuracy rows are the fixed 512-row subsets for NASBench101, ENAS, and NASNet from `GraphArch-Regression` revision `c557392740094b539bbdb527d03e3a78e5b34a38`. Each serialized ONNX input is truncated only at the author-card 4,096-token limit; generation uses the released numeric decoder, eight stochastic draws, and the median prediction.

The accuracy alias and the Table-3 memory/latency alias share every inference-critical file. The common `model.safetensors` is 725,864,700 bytes with SHA-256 `7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c`. This establishes that one checkpoint—not separately trained task models—produced all Claim-1 metrics.

Independent verification recomputed all 1,536 stored medians, per-space Spearman correlations, bootstrap intervals, a three-space label-permutation null, and the input-identity shuffle control. It also re-hashed the local released checkpoint and requires the complete local parquet to match its Hub LFS digest.

Environment: Python 3.12; transformers 4.53.2 (the checkpoint export version); PyTorch 2.7.1+cu126; NVIDIA GTX 1050; float16 inference; deterministic batch seeds; raw draws checkpointed after every four-row batch. Tests and publication gates fail closed on scale, space coverage, checkpoint provenance, route count, pinned-cell count, and leaked local paths.

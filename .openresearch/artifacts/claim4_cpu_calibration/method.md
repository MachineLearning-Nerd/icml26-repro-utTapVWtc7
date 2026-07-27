# Claim 4 CPU calibration

This node downloads the pinned 4.116 GB GraphArch Parquet, verifies its SHA-256,
and scans it sequentially with bounded memory. The full-file path is required
because the Hugging Face dataset filter endpoint indexes only a partial prefix
that excludes DARTS.

It then runs the exact released DARTS checkpoint and eight-draw median decoding
path on 16 deterministic released rows. It records setup and inference time
separately, model identity, raw draws, two Kendall implementations, and a
200-permutation shuffled-target control.

The run is deliberately labeled calibration-only. The small sample is not
Table 4 evidence and cannot change Claim 4 from `BLOCKED`. Its purpose is to
size the subsequent 512-row public-checkpoint evaluation from observed CPU
throughput.

Exact inherited command:

```bash
uv run --locked python repro/src/run_campaign.py
```

Estimated active CPU requirement: 8 cores plus bounded single-threaded Parquet
decoding. Selected compute: Hugging Face `cpu-upgrade`; no GPU.

# Method

Run exactly:

```bash
uv run --locked python repro/src/run_campaign.py
```

The entrypoint reads only `repro/config/campaign.json`, prints the Git SHA,
estimated and allocated CPU counts, selected backend/flavor, and elapsed time,
then independently recomputes accepted statistics from committed raw evidence.
BLAS/OpenMP-backed calculations are limited to the committed one-core estimate.

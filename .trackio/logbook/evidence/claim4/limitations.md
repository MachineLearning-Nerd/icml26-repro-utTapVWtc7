# Claim 4 limitations and deviations

- Only Amoeba, DARTS, and PNAS target-specific RLM checkpoints are public.
- The public target-specific repositories have no model cards and no
  training-row, evaluation-row, seed, or trainer manifests.
- The RLM repository does not include the Table 4 training/evaluation setup.
- The dataset card documents a broad fine-tuning strategy but omits the
  information needed to reconstruct exact model states or held-out rows.
- The retained 512-row ENAS and NASNet results use the unified base model and
  are metric-correction diagnostics only.
- The audit neither verifies nor falsifies the exact five-space claim.

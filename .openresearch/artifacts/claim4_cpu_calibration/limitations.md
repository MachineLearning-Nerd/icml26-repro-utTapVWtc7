# Calibration limitations

- Sixteen rows are not paper scale.
- The unknown DARTS fine-tuning rows are not excluded.
- Kendall on sixteen rows has high sampling uncertainty.
- Linear runtime projection may miss sequence-length and cache effects.
- The release stores all 611,931 examples in one Parquet row group, so DARTS
  access requires downloading the complete 4.116 GB file.
- No scientific points or confidence are assigned from this run.

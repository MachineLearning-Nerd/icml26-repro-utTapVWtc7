# Limitations and deviations

- The judged repository retains full row-level draws for three ONNX accuracy
  spaces and 17-language CodeNet predictions, but only paper-scale summaries
  for APPS and KBSS. The baseline therefore recomputes the former and
  integrity-checks the latter.
- The baseline does not claim that retained evidence was regenerated.
- The original evidence used GPU runs; this campaign is CPU-only. Historical
  hardware provenance is preserved, not rewritten.

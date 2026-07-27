# Baseline evaluation

Expected verdict: **PASS**, preserving the live judge's full credit for Claims
1–3. The OpenResearch run log is authoritative for the actual rerun result.

The root's first run correctly failed on a provenance mismatch. This child
separately recomputes the primary Colab CodeNet bundle (mean 0.529850) and the
independent CPU CSV (mean 0.523403); neither is substituted for the other.

This is an integrity and independent-statistics baseline, not a new scientific
claim. Full model inference is intentionally outside this short frozen-root
check and must occur on child branches.

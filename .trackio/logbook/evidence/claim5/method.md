# Claim 5 method

The fixed campaign command is:

```bash
uv run --locked python repro/src/run_campaign.py
```

The committed phase `claim5_finalize` performs a public-artifact audit and
reruns the accepted Claims 1–3 plus the final Claim 4 integrity checker. It:

1. recursively audits pinned paper-time and current official `regress-lm`
   repository trees, including the contents of every small text source/config
   file, for exact ablation/scaling markers;
2. audits the author's complete public Hugging Face model listing for a
   Table 5 model or a b-b/600M RLM;
3. verifies pinned metadata and serialized parameter counts for the two named
   T5Gemma prefix-LM bases;
4. pins and hashes the cited Qin et al. implementation and checks the exact
   normalized-target/scalar-regressor pattern;
5. uses a synthetic exact-artifact filename as a detector negative control;
6. requires four distinct routes, including a dedicated falsification route;
7. invokes `repro/src/claim5_exact_gate.py` and requires its nonzero,
   fail-closed result.

Network requests use an explicit User-Agent. Dynamic response hashes and exact
revisions are printed in the raw result. The verifier exits nonzero on an
unexpected upstream artifact or metadata change so that it cannot silently
preserve the BLOCKED verdict after new evidence appears.

This audit is expected to use one active CPU core and finish in under five
minutes, so the authorized backend is local CPU. It trains no model and makes
no performance forecast from repository absence.

# Claim 4 source audit

Paper source: <https://ar5iv.labs.arxiv.org/html/2509.26476>, retrieved
2026-07-27 with user agent `OpenResearch-Reproduction/1.0`. The retrieved HTML
SHA-256 is
`5947f4512cc86850a63409adf52af25ac1f40b15dcc797348fd8ae91a2740913`.

The exact RLM statement is Table 4 (`S5.T4`) and Section 5.3
(`S5.SS2`). The table reports Kendall tau over five NDS search spaces:
NASNet 0.382, Amoeba 0.488, PNAS 0.427, ENAS 0.481, and DARTS 0.528.
Their unrounded arithmetic mean is 0.4612, displayed as 0.461. Reported
baseline averages are Arch2Vec 0.212, CATE 0.238, GNN 0.429, and FLAN
0.459.

The Table 4 caption specifies 16 target examples for NASNet, Amoeba, and
PNAS and 100 for DARTS, but omits ENAS. The cited FLAN primary source,
Akhauri and Abdelfattah, *Encodings for Prediction-based Neural Architecture
Search*, arXiv:2403.02484, resolves the inherited protocol:

- Section 5 states that a base predictor is trained on 1,024 source examples.
- FLAN Table 5 uses 16 target examples for all five source-target transfers.
- It evaluates the predictor on the remaining full target search space and
  averages three trials.
- The exact FLAN values copied into the RLM table are the 16-shot values,
  except RLM explicitly changes DARTS to 100 shots.

The author GraphArch dataset card at revision
`c557392740094b539bbdb527d03e3a78e5b34a38` says the authors' preferred RLM
adaptation uses about 1,024 examples from three related NDS spaces, 16 target
examples repeated eight times, random shuffling, and learning rate 1e-4 with
cosine decay. It does not provide epochs, batch size, seeds, row identifiers,
or the exact per-target source composition.

The public FLAN repository was inspected at commit
`1a9d282e9934c9e660834f495caa6a74719f262c`. Its
`correlation_trainer/universal_main.py` defaults to three trials, samples
target rows without replacement, and tests on the remaining target rows.

The public RLM implementation repository was inspected at commit
`6c23ccb51ae9862b03e9aca1a2ea936b6fe56f843`. It contains reusable model
components but no Table 4 training command, committed target split, or
evaluation configuration.

The source statement is an empirical finite-domain claim. Full verification
requires all five target-specific model states or an exact reproducible
training recipe, plus the identities of the fine-tuning and held-out rows.
The unified base model cannot stand in for a missing few-shot checkpoint.

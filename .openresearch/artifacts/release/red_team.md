# Evaluator-blind pre-publication red team

## Round 1 — rejected

The review began from a fresh checkout of the exact judged Space revision
`19231479d69a31c8e01832c24e146c37eca9a5ba`. Comparing the protected manifest
to the first overlaid candidate showed that five historical judged Markdown
pages had been edited in place. Although their paths remained present, this
violated the additive historical-evidence contract.

Fix: the exact judged bytes were restored at all original page paths. Current
Claim 1–3 and index content moved to new `current-*` pages. `README.md` and
`logbook.json` now put current verification first while keeping every judged
page reachable.

## Round 2 — PASS

The second review received only the candidate and started at `README.md`. A
generic relative-link crawler followed every link and the `logbook.json`
navigation without repository knowledge. It opened 56 files; the exact list is
in `evaluator_blind_review.json`.

Results:

- all five current claim pages were reachable and showed their exact verdict;
- the visibility matrix contained all five rows;
- both exact scientific gates, both final checkers, raw Claim 4/5 outputs, and
  the locked environment were reachable;
- every relative link resolved;
- the 21-file judged file set was a subset of the candidate;
- all seven protected historical judged pages retained their exact SHA-256;
- the 65-path upload allowlist exactly matched modified/untracked candidate
  paths;
- all 64 non-self manifest hashes verified;
- no token/private-key signature was detected in any upload path.

The reviewer could not independently recover full-scale APPS/KBSS row-level
draws because they were not retained in the judged repository. The current
Claim 2 page states this limitation and links the accepted paper-scale aggregate
plus the historical n=40 raw control. No new point depends on that gap.

## Round 3 — PASS after standalone-execution repair

An upload preflight was rejected by Hugging Face's request-rate limit before
any Space commit was created. During the retry audit, the reviewer downloaded
the candidate and tried to execute the copied checkers from the Space root.
That exposed a path-contract defect: the first copies still addressed internal
`.openresearch/` repository paths, so visible source was not independently
executable from the evaluator artifact.

Fix: the current Space copies now address only `evidence/` paths. A new
standard-library-only `code/verify_cumulative_space.py` recomputes the 1,536
ONNX rows and 3,400 independent CodeNet rows and checks the retained APPS/KBSS
summary and hashes. `code/verify_space_release.py` runs that checker plus both
Claim 4/5 integrity checkers. The candidate audit executes all of them and
separately requires both exact scientific gates to exit nonzero with their
documented BLOCKED reason.

The third blind traversal started again at `README.md` from a fresh checkout of
the exact judged revision. All links, protected-history hashes, manifest
hashes, allowlist paths, claim pages, raw records, and environment links pass.
The standalone release checker exits 0; the Claim 4 and Claim 5 exact gates
each exit 1 for the intended missing-artifact reason. The APPS/KBSS row-level
limitation remains the only conclusion the blind reviewer cannot independently
recompute, and remains explicit on the current page.

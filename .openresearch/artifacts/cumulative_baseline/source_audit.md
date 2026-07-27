# Source audit — frozen Claims 1–3 baseline

- Paper HTML: `https://ar5iv.labs.arxiv.org/html/2509.26476`
- Retrieval: `2026-07-27T11:45:53Z`, explicit browser User-Agent
- HTML SHA-256: `5947f4512cc86850a63409adf52af25ac1f40b15dcc797348fd8ae91a2740913`
- Live verdict dataset: `https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts/resolve/main/verdicts.json`
- Retrieval: `2026-07-27T11:46:17Z`
- Verdict dataset SHA-256: `3c8514624f8c13fcfa5a2dfe68d9aced5eb57f85070309270232eab628798065`
- Filter: exact `space_id == "DineshAI/utTapVWtc7"`
- Judged Space revision: `19231479d69a31c8e01832c24e146c37eca9a5ba`

The baseline contract deliberately preserves the three claims already awarded
full credit. It does not reinterpret them or use their formulae to choose a
sample size. It recomputes metrics on the previously retained paper-scale rows
and fails closed on their hashes and structure.

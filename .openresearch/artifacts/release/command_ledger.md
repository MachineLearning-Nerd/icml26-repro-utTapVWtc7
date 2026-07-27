# Campaign command ledger

All formal experiment nodes inherited the exact command:

```bash
uv run --locked python repro/src/run_campaign.py
```

Startup and source audit commands:

```bash
orx skill
orx skill orx-experiment-tree
orx skill orx-evidence
orx skill orx-git
orx skill orx-compute
orx projects --json
orx runs dd9dbc02-d010-4153-b8d4-79e93526cbae
git branch -a
git status --short
git rev-parse HEAD
df -h
curl -A 'OpenResearch-Reproduction/1.0 paper-2509.26476' https://ar5iv.labs.arxiv.org/html/2509.26476
orx paper 2509.26476 --full
```

Experiment launches, in tree order:

```bash
orx exp run b9e6af49-17d9-4515-8ef4-4942484eaea7 --backend local
orx exp run 1efb213a-0a2a-4f25-a097-290e3d5eb98d --backend hf --flavor cpu-upgrade
orx exp run 53009529-0ddf-4e87-a406-fc1c92c2650c --backend hf --flavor cpu-upgrade
orx exp run d31e52ca-c363-422f-b24e-48bd6e34d22f --backend hf --flavor cpu-upgrade
orx exp run 0e9fe13a-d6d5-4cda-87ee-62908e163026 --backend hf --flavor cpu-upgrade
orx exp run b466910b-0e17-439d-b565-d7432bf35248 --backend hf --flavor cpu-upgrade
orx exp run dc608184-d764-49a7-ad82-3fef3dad0e24 --backend hf --flavor cpu-upgrade
orx exp run b5d71f8c-7218-434b-a20b-fc616f170ad3 --backend local
orx exp run 31c2cd5c-f6c7-45bd-b42f-c2a84339d768 --backend local
orx exp run 9457deba-32cb-4c9a-af40-a094df7bfa0b --backend local
orx exp run 2b60a341-625c-4b19-81c9-fd9ef67876e2 --backend local
orx exp run ea036db1-280c-472b-8b6a-4ea62599296c --backend local
orx exp run 040fdc5f-0eb8-47f3-906e-aadb7f3271a9 --backend local
orx exp run c5179780-c2a8-4b87-a746-dc7541fec474 --backend local
```

Every run was monitored with `orx exp wait <experiment-id> --timeout 480`,
enumerated with `orx runs dd9dbc02-d010-4153-b8d4-79e93526cbae`, and analyzed
with `orx logs <run-id>`. Branch changes used `git fetch origin`,
`git checkout <orx-branch>`, scoped `git add`, `git commit`, and
`git push origin HEAD`.

Release validation commands:

```bash
uv run --locked marimo check --strict notebooks/regresslm_reproduction.py
uv run --locked python repro/src/build_report_figures.py reports/regression-language-models-code-2026-07-27/images
uv run --locked python repro/src/verify_candidate_space.py /tmp/uttap-space-candidate.NpTEnt .openresearch/artifacts/protected_judged_space_manifest.sha256
python code/verify_space_release.py
uv run --locked python repro/src/verify_candidate_space.py /tmp/uttap-space-candidate-round3.BwTOvk .openresearch/artifacts/protected_judged_space_manifest.sha256
```

Publication and post-publication commands:

```bash
git ls-remote https://huggingface.co/spaces/DineshAI/utTapVWtc7 refs/heads/main
uv run --locked python  # HfApi.create_commit: repo_type=space, 67 text operations, parent_commit=19231479d69a31c8e01832c24e146c37eca9a5ba
git clone https://huggingface.co/spaces/DineshAI/utTapVWtc7 /tmp/uttap-space-published.010cw6
git -C /tmp/uttap-space-published.010cw6 checkout f791e670006b5d428a2e11593eb29fb568bb4b45
shasum -a 256 -c MANIFEST.sha256
python3 code/verify_space_release.py
python3 code/claim4_exact_gate.py
python3 code/claim5_exact_gate.py
git push origin 700aa77c312993386e9ff4036abbc94a9856898b:refs/heads/master
git push origin HEAD:refs/heads/master
git ls-remote origin refs/heads/master
curl -fsSL -A 'OpenResearch-Reproduction/1.0 paper-2509.26476 post-publication' https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts/resolve/main/verdicts.json
```

The API call used the cached Hugging Face authentication context without
printing credentials. It returned published revision
`f791e670006b5d428a2e11593eb29fb568bb4b45`.

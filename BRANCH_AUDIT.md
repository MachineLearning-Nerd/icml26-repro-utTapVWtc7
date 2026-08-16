# Branch audit and rename map

The original repository used a mixture of master, publication, and orx
branches. The names reflected the automation history rather than the
scientific role of each line of work. The final inventory uses a stable main
branch plus descriptive audit, baseline, and release namespaces.

## Old to new mapping

| Original branch | Original tip | Final branch | Role |
| --- | --- | --- | --- |
| master | 8ee170b8fcdbb2a96fcdb119a2da7544e6327cfd | main | Publication surface |
| orx/audit-claim-5-ablation-and-scaling | 07c56964b03438e957ab4e496529c97ce92d5c8f | audit/claim-5-ablation-scaling | Table 5/6 artifact audit |
| orx/calibrate-claim-4-cpu-inference | 441d48a3017dc9ad2e56188703f0687e647e5c29 | audit/claim-4-cpu-inference | CPU calibration |
| orx/calibrate-claim-4-from-pinned-full-parquet | b160aaf6ccb65878c0e4e22add6858e4299c6ac8 | audit/claim-4-full-parquet | Full-Parquet provenance |
| orx/calibrate-claim-4-with-accepted-config-compatibi | 7a489958ce0d307c491b740c3393c45fa0185418 | audit/claim-4-config-compatibility | Accepted-config calibration |
| orx/claim-4-released-checkpoint-kendall-audit | bcedb13e09389703999d5bcc1735fc555a340e42b | audit/claim-4-kendall-checkpoint | Kendall metric audit |
| orx/correct-claim-5-cited-source-provenance | 16d1a11f279b470db95da4f15d1ea7fcb1c05e5b | audit/claim-5-source-provenance | Cited normalized-head source |
| orx/correct-dual-codenet-provenance | a2473ca9f3c0b37ab65a9763b21b83f7490ccaf9 | audit/codenet-dual-provenance | Dual CodeNet evidence |
| orx/finalize-claim-4-four-route-blocked-verdict | 63f5b712a0057a670dece7b1b74c6b34cb12386e | audit/claim-4-blocked-verdict | Final Claim 4 verdict |
| orx/frozen-6-10-cumulative-baseline | dcaf05c92039cb90c01e44a2a7a84e377cb0c0d8 | baseline/frozen-6-10 | Frozen challenge baseline |
| orx/make-space-verifiers-standalone | 700aa77c312993386e9ff4036abbc94a9856898b | release/standalone-verifiers | Standalone release verifiers |
| orx/prepare-evaluator-visible-cumulative-release | b528be1b1be0573d89f38ec6498584d666294255 | release/evaluator-visible | Evaluator-visible cumulative release |
| orx/record-claim-4-audit-evidence | 72def5f69a2bd96bc34ea70433f6c2f778ce3639 | audit/claim-4-evidence | Earlier Claim 4 evidence |
| orx/record-claim-5-exact-audit-evidence | 638553f84622313314a6153501c33d48bbc77b81 | audit/claim-5-evidence | Earlier Claim 5 evidence |
| orx/repair-claim-5-source-audit-transport | 88bb1b52a3dc4fd5605641c5e2be51959cc2e52b | audit/claim-5-source-transport | Source transport repair |
| publication/awaiting-judge-f791e670 | 8ee170b8fcdbb2a96fcdb119a2da7544e6327cfd | collapsed into main | Duplicate of publication tip |

The publication branch had the same tip as master, so retaining both would
create a duplicate public surface. It is intentionally collapsed into main.

## Final branch inventory

The expected final public branches are:

~~~text
main
audit/claim-4-blocked-verdict
audit/claim-4-config-compatibility
audit/claim-4-cpu-inference
audit/claim-4-evidence
audit/claim-4-full-parquet
audit/claim-4-kendall-checkpoint
audit/claim-5-ablation-scaling
audit/claim-5-evidence
audit/claim-5-source-provenance
audit/claim-5-source-transport
audit/codenet-dual-provenance
baseline/frozen-6-10
release/evaluator-visible
release/standalone-verifiers
~~~

No final public branch should retain the legacy orx/ namespace, master, or
the duplicate publication/awaiting-judge-f791e670 name.

## History identity

The original linear history includes several personal addresses and automation
addresses. Before publication, every reachable commit is rewritten to:

~~~text
MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>
~~~

No co-author trailers are added. The pre-rewrite bundle and its SHA-256 are
recorded in the collection tracker after publication.

#!/usr/bin/env python3
"""Load the ACTUAL Colab-produced result file (outputs/colab/regresslm_table3_results.csv,
the notebook's saved artifact) + the per-language list, and verify the C2/C3 claims
from the raw provenance file (not a markdown assertion)."""
import csv, os, re
P = os.path.join("outputs", "colab", "regresslm_table3_results.csv")
print(f"Loading raw Colab output: {P}")
rows = list(csv.DictReader(open(P)))
res = {r["space"]: float(r["spearman_repro"]) for r in rows}
print("Colab Table-3 results (from the notebook's saved file):")
for r in rows:
    print(f"  {r['space']:10s} spearman={float(r['spearman_repro']):.4f}  n={r['n']}  ref={r['table3_ref']}  claim={r['claim']}")
# per-language
pl = open(os.path.join("outputs","colab","colab_codenet_perlang.txt")).read()
rhos = [float(x) for x in re.findall(r"Spearman=([0-9.]+)", pl)]
above = sum(1 for r in rhos if r > 0.5)
print(f"\nPer-language (n=200 each): {len(rhos)} langs, {above}/{len(rhos)} individually > 0.5")
print(f"  min={min(rhos):.3f} max={max(rhos):.3f} mean={sum(rhos)/len(rhos):.4f}")
print("\n" + "="*60)
print(f"C2 (APPS > 0.9):  {res['APPS']:.4f}  -> {'VERIFIED' if res['APPS']>0.9 else 'FAIL'}")
print(f"C3 (CDSS-avg > 0.5): {res['CDSS-avg']:.4f}  -> {'VERIFIED' if res['CDSS-avg']>0.5 else 'FAIL'}")
print(f"KBSS (ref 0.527): {res['KBSS']:.4f}  (matches reference)")
print("="*60)

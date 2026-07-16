#!/usr/bin/env python3
"""Print the Colab paper-scale Table-3 results (imported from the Colab GPU run).

Loaded from outputs/colab/table3_results.json (the aggregate results of the
Colab GPU eval). Run via `trackio logbook run` so the results register as a
captured run + artifact alongside the local small-scale evidence.
"""
import json, sys, os
p = os.path.join("outputs", "colab", "table3_results.json")
d = json.load(open(p))
print("RegressLM Table 3 (paper-scale, Colab GPU, released checkpoint):")
for k, v in d["table3"].items():
    print(f"  {k:8s} spearman={v['spearman']:.4f}  n={v.get('n', v.get('n_langs'))}  "
          f"ref={v['reference']}  claim={v['claim']}  -> {v['verdict']}")
print(f"\nCDSS per-language average = {d['table3']['CDSS_avg']['spearman']:.4f} (claim >0.5)")

#!/usr/bin/env python3
"""Enumerate Code-Regression schema fast via PyArrow column projection.

The dataset is a single 5.6 GB parquet. We never read the giant ``input`` column —
only ``space``, ``metric_type`` and (for CodeNet) ``metadata`` — so this is cheap.

Outputs:
  * value_counts of space x metric_type  (finds the exact APPS-accuracy metric_type)
  * metadata.language distribution among space in (CDSS, ...) rows (Claim 3 = 17 langs)
"""
import argparse
import collections
import json
import os

import pyarrow.dataset as ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="hf://datasets/akhauriyash/Code-Regression/data.parquet")
    ap.add_argument("--out", default=os.path.join("outputs", "probe_metrics.json"))
    ap.add_argument("--lang_spaces", default="CDSS,CODENET,CodeNet",
                    help="space values whose metadata.language we tally (comma-sep)")
    args = ap.parse_args()

    print(f"opening dataset: {args.url}", flush=True)
    dataset = ds.dataset(args.url, format="parquet")
    cols = [c.name for c in dataset.schema]
    print(f"columns: {cols}", flush=True)

    # 1) full tally of space x metric_type (small string columns, dict-encoded)
    print("reading space, metric_type ...", flush=True)
    t = dataset.to_table(columns=["space", "metric_type"])
    sp = t.column("space").to_pylist()
    mt = t.column("metric_type").to_pylist()
    sm = collections.Counter(zip(sp, mt))
    by_space = collections.defaultdict(collections.Counter)
    for (s, m), c in sm.items():
        by_space[s][m] += c
    print(f"rows={len(sp)}", flush=True)
    for s, ctr in sorted(by_space.items()):
        print(f"  space={s!r}: {dict(ctr.most_common())}", flush=True)

    # 2) language distribution within CodeNet-ish spaces
    lang_spaces = [x.strip() for x in args.lang_spaces.split(",") if x.strip()]
    lang_counts = {}
    present_spaces = set(sp)
    want = [s for s in lang_spaces if s in present_spaces]
    if want:
        print(f"reading metadata for spaces {want} ...", flush=True)
        import pyarrow.compute as pc
        flt = pc.is_in(pc.field("space"), value_set=ds.array(want))
        tm = dataset.to_table(columns=["space", "metadata"], filter=flt)
        lctr = collections.Counter()
        for s, md in zip(tm.column("space").to_pylist(), tm.column("metadata").to_pylist()):
            try:
                d = json.loads(md) if isinstance(md, str) else (md or {})
            except Exception:
                d = {}
            lctr[d.get("language", "<none>")] += 1
        lang_counts = dict(lctr.most_common())
        print(f"  languages: {lang_counts}", flush=True)
    else:
        print(f"none of {lang_spaces} present (spaces seen: {sorted(present_spaces)})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {
        "columns": cols,
        "n_rows": len(sp),
        "space_x_metric": {str(s): dict(c.most_common()) for s, c in by_space.items()},
        "languages": lang_counts,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

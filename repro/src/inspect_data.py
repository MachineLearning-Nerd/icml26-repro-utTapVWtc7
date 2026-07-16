#!/usr/bin/env python3
"""Inspect Code-Regression cheaply with PyArrow (filter pushdown + early stop).

No ``datasets`` streaming (slow over the 5.6 GB single-file parquet). We open the
parquet once, then:
  * PEEK: scanner(columns incl input, filter=space).head(n) — stops after n matches
  * LANGS: scanner(columns=[metadata], filter=space=CDSS).to_batches() — metadata only
"""
import argparse
import collections
import json
import os
from ast import literal_eval

import pyarrow as pa
import pyarrow.dataset as ds

URL = "hf://datasets/akhauriyash/Code-Regression/data.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peek_per_space", type=int, default=3)
    ap.add_argument("--lang_scan_rows", type=int, default=60_000)
    ap.add_argument("--out", default=os.path.join("outputs", "inspect_data.json"))
    args = ap.parse_args()

    print(f"opening {URL} ...", flush=True)
    dataset = ds.dataset(URL, format="parquet")

    print("\n=== PEEK (input head=80) ===", flush=True)
    peek = {}
    for space in ["APPS", "KBSS", "CDSS"]:
        flt = pa.compute.equal(pa.compute.field("space"), space)
        sc = dataset.scanner(
            columns=["space", "input", "target", "metric_type", "metadata"],
            filter=flt, batch_size=64)
        rows = []
        d = sc.head(args.peek_per_space).to_pydict()
        n = len(d.get("space", []))
        for i in range(n):
            md = d["metadata"][i]
            try:
                mdd = literal_eval(md) if isinstance(md, str) else {}
            except Exception:
                mdd = {}
            print(f"  [{space}] target={d['target'][i]} metric_type={d['metric_type'][i]} "
                  f"lang={mdd.get('language')} | input={str(d['input'][i])[:80]!r}", flush=True)
            rows.append(dict(space=space, target=d["target"][i],
                             metric_type=d["metric_type"][i], lang=mdd.get("language"),
                             input_head=str(d["input"][i])[:80]))
        peek[space] = rows

    print(f"\n=== CDSS language tally (<= {args.lang_scan_rows} rows, metadata col only) ===", flush=True)
    flt = pa.compute.equal(pa.compute.field("space"), "CDSS")
    lang = collections.Counter()
    scanned = 0
    for batch in dataset.scanner(columns=["metadata"], filter=flt,
                                 batch_size=65536).to_batches():
        for md in batch.column("metadata"):
            try:
                dd = literal_eval(md.as_py()) if md.is_valid else {}
            except Exception:
                dd = {}
            lang[dd.get("language", "<none>")] += 1
            scanned += 1
        if scanned >= args.lang_scan_rows:
            break
    print(f"scanned {scanned} CDSS rows; distinct languages={len(lang)}", flush=True)
    for lg, c in lang.most_common(30):
        print(f"  {lg!r}: {c}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(peek=peek, cdss_scanned=scanned,
                       cdss_languages=dict(lang.most_common())), fh, indent=2, default=str)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

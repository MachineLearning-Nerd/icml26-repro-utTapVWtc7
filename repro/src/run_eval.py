#!/usr/bin/env python3
"""RegressLM Table-3 reproduction evaluator.

Implements the authors' published eval recipe (Code-Regression dataset card),
which produced:  KBSS=0.527  CDSS=0.787  APPS=0.926  for the released
``RegressLM-gemma-s-RLM-table3`` checkpoint.

Recipe (faithful):
  * input prefix per space:
      APPS/KBSS ->  "{SPACE}\\n{input}"
      CDSS      ->  "# CDSS\\n# Language: {lang}\\n{input}"
  * generate(do_sample=True, top_p=0.95, temperature=1.0,
            min_new_tokens=max_new_tokens=N_OUT)  where N_OUT = num_tokens_per_obj*max_num_objs
  * decode tok.token_ids_to_floats(seq)[0] per sample; aggregate samples by median
  * Spearman(targets, preds)

Rows are fetched with PyArrow filter pushdown + early stop (the parquet is a single
5.6 GB file; ``datasets`` streaming is far too slow to find a handful of rows).
Phase A = local CPU, small N. Phase B = Colab GPU, full N (notebook).
"""
import argparse
import csv
import json
import math
import os
from ast import literal_eval

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from scipy import stats

REPO = "akhauriyash/RegressLM-gemma-s-RLM-table3"
PARQUET = "hf://datasets/akhauriyash/Code-Regression/data.parquet"
_LOCAL_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "rlm-table3")


def pick_device(arg):
    """Use CUDA only if a kernel actually runs (torch cu130 has no sm_61 kernels
    for the GTX 1050, though cuda.is_available() is still True)."""
    if arg:
        return arg
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception as e:
            print(f"CUDA present but unusable ({type(e).__name__}); falling back to CPU", flush=True)
    return "cpu"


def load_model(device, dtype, path=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if path:
        src = path
    elif os.path.isdir(_LOCAL_CKPT):
        src = os.path.abspath(_LOCAL_CKPT)
    else:
        # Load from the LOCAL snapshot dir, not the repo-id. The P10 tokenizer only
        # uses the bundled encoder_tokenizer/ subdir when Path(model_path).exists();
        # a repo-id string makes it fall back to the GATED google/t5gemma-s-s-prefixlm
        # for the encoder vocab -> 401 GatedRepoError.
        from huggingface_hub import snapshot_download
        src = snapshot_download(REPO)
    print(f"loading model from: {src}", flush=True)
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        src, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    return tok, model


def fetch_rows(space, lang=None, limit=512, seed=42):
    """First `limit` rows of `space` (optionally metadata.language==lang),
    formatted with the per-space prefix the model expects. Returns (inputs, targets)."""
    dataset = ds.dataset(PARQUET, format="parquet")
    flt = pa.compute.equal(pa.compute.field("space"), space)
    cols = ["input", "target", "metadata"]
    inputs, targets = [], []
    # Small batch_size so the scanner yields quickly and we break at `limit`
    # (the input column is huge; batch_size=2048 pulled whole row groups over HTTP).
    for batch in dataset.scanner(columns=cols, filter=flt, batch_size=256).to_batches():
        for inp, tgt, md in zip(batch.column("input"), batch.column("target"),
                                batch.column("metadata")):
            try:
                mdd = literal_eval(md.as_py()) if md.is_valid else {}
            except Exception:
                mdd = {}
            if lang is not None and mdd.get("language") != lang:
                continue
            if tgt.is_valid is False or inp.is_valid is False:
                continue
            x = inp.as_py()
            if space == "CDSS":
                x = f"# {space}\n# Language: {mdd.get('language')}\n{x}"
            else:
                x = f"{space}\n{x}"
            inputs.append(x)
            targets.append(float(tgt.as_py()))
            if len(inputs) >= limit:
                return inputs, targets
    return inputs, targets


def decode_seq(tok, ids):
    fs = tok.token_ids_to_floats(ids)
    if isinstance(fs, (list, tuple)):
        return float(fs[0]) if fs else math.nan
    return float(fs)


@torch.inference_mode()
def predict(tok, model, inputs, num_samples, device, n_out, batch_size, dtype):
    """Return one median prediction per input (num_samples draws each)."""
    preds = [math.nan] * len(inputs)
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True,
                  max_length=2048).to(device)
        draws = []
        for _ in range(num_samples):
            out = model.generate(
                **enc, do_sample=True, top_p=0.95, temperature=1.0,
                min_new_tokens=n_out, max_new_tokens=n_out,
                pad_token_id=getattr(tok, "pad_token_id", 0),
                use_cache=True,
            )
            seqs = out.view(enc["input_ids"].shape[0], -1) if out.dim() == 2 else out
            vals = []
            for r in range(seqs.shape[0]):
                ids = seqs[r].tolist()
                try:
                    vals.append(decode_seq(tok, ids))
                except Exception:
                    vals.append(math.nan)
            draws.append(vals)
        med = np.nanmedian(np.array(draws, dtype=float), axis=0)
        for j, v in enumerate(med):
            preds[i + j] = float(v)
        print(f"    batch {i//batch_size + 1}/{(len(inputs)+batch_size-1)//batch_size} "
              f"done ({i+len(chunk)}/{len(inputs)})", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", default="APPS")
    ap.add_argument("--lang", default=None, help="metadata.language (CDSS only)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    # FP16 is supported by both the local Pascal GPU and Colab T4/L4.  Pascal
    # has no native BF16, and emulation is slower and less predictable.
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    tok, model = load_model(device, dtype, args.model_path)
    n_out = int(getattr(model.config, "num_tokens_per_obj", 8)) * \
            int(getattr(model.config, "max_num_objs", 1))
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | n_out_tokens={n_out}",
          flush=True)

    print(f"fetching space={args.space} lang={args.lang} limit={args.limit} ...", flush=True)
    inputs, targets = fetch_rows(args.space, args.lang, args.limit, args.seed)
    if not inputs:
        print("NO ROWS MATCHED — check --space/--lang."); raise SystemExit(2)
    yt = np.array(targets, dtype=float)
    print(f"got {len(inputs)} rows; target range [{np.nanmin(yt):.4f}, {np.nanmax(yt):.4f}] "
          f"mean={np.nanmean(yt):.4f}", flush=True)

    print(f"predicting (num_samples={args.num_samples}, batch={args.batch_size}) ...", flush=True)
    yp = np.array(predict(tok, model, inputs, args.num_samples, device, n_out,
                          args.batch_size, dtype), dtype=float)
    v = np.isfinite(yt) & np.isfinite(yp)
    res = dict(
        space=args.space, lang=args.lang, n=int(v.sum()), n_total=len(yt),
        num_samples=args.num_samples, batch_size=args.batch_size, seed=args.seed,
        device=device,
        spearman=float(stats.spearmanr(yt[v], yp[v]).correlation),
        pearson=float(stats.pearsonr(yt[v], yp[v])[0]),
        mse=float(np.mean((yt[v] - yp[v]) ** 2)),
        nan_rate=float(1 - v.mean()),
    )
    print("=" * 60); print(json.dumps(res, indent=2)); print("=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["i", "y_true", "y_pred"])
        for i in range(len(yt)):
            w.writerow([i, yt[i], yp[i]])
    with open(args.out.replace(".csv", ".json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

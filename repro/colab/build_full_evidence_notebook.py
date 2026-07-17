#!/usr/bin/env python3
"""Generate the one-click Colab evidence notebook without notebook dependencies."""

import json
from pathlib import Path


OUT = Path(__file__).with_name("regresslm_full_evidence_colab.ipynb")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md("""# RegressLM — missing-evidence repair (Colab GPU)

This notebook produces the two pieces the official judge found missing:

1. **Claim 1 / ONNX accuracy:** the same released unified checkpoint is evaluated on released `GraphArch-Regression` ONNX strings with `val_accuracy` targets.
2. **Claim 3 / execution provenance:** all 17 scored CodeNet languages are evaluated at 200 rows/language with visible model-loading, fetch, batch-progress, timing, seed, and environment logs.

Use a **T4, L4, or A100 GPU runtime**, then choose **Runtime → Run all**. The final cell downloads `regresslm_evidence_bundle.zip`; send that ZIP back to Codex. APPS and KBSS are disabled by default because the official judge already accepted their n=512 results, but one switch below enables them.
"""),
    code("""# Configuration: the defaults directly target the two rejected claims.
RUN_CODENET_17 = True
RUN_ONNX_ACCURACY = True
RUN_ALREADY_ACCEPTED_APPS_KBSS = False

ROWS_PER_LANGUAGE = 200
ONNX_ROWS_PER_SPACE = 64
ONNX_SPACES = ["NASBench101", "ENAS", "NASNet"]
NUM_SAMPLES = 8
BATCH_SIZE = 16
SEED = 42
print({k: v for k, v in globals().items() if k.startswith("RUN_") or k in {
    "ROWS_PER_LANGUAGE", "ONNX_ROWS_PER_SPACE", "ONNX_SPACES", "NUM_SAMPLES", "BATCH_SIZE", "SEED"
}})
"""),
    code("""import os, sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "regress-lm[extras]", "pyarrow", "scipy", "pandas", "requests"], check=True)
# The checkpoint declares this exact version. Transformers 5.x makes this custom
# encoder-decoder input-insensitive, so install 4.53.2 last.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.53.2"], check=True)

import csv, glob, hashlib, json, math, platform, random, shutil, time, urllib.request, zipfile
from ast import literal_eval
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.dataset as ds, requests, torch
from scipy import stats
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

assert torch.cuda.is_available(), "Select a GPU runtime: Runtime > Change runtime type > T4/L4/A100"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = "cuda"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
RESULTS = Path("regresslm_evidence"); RESULTS.mkdir(exist_ok=True)
environment = {
    "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "transformers": __import__("transformers").__version__, "dtype": str(dtype), "seed": SEED,
}
print(json.dumps(environment, indent=2), flush=True)
"""),
    code("""# Load the public checkpoint locally so its bundled encoder tokenizer is used.
from huggingface_hub import snapshot_download, hf_hub_download
REPO = "akhauriyash/RegressLM-gemma-s-RLM-table3"
CKPT_DIR = snapshot_download(REPO)
PATCH_RAW = "https://raw.githubusercontent.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7/master/repro/patches"
for fn in ["configuration_regresslm.py", "modeling_regresslm.py"]:
    urllib.request.urlretrieve(f"{PATCH_RAW}/{fn}", os.path.join(CKPT_DIR, fn))
for d in glob.glob(os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/*")):
    if os.path.exists(os.path.join(d, "configuration_regresslm.py")):
        shutil.rmtree(d, ignore_errors=True)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    CKPT_DIR, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()
N_OUT = int(model.config.num_tokens_per_obj) * int(model.config.max_num_objs)
model_info = {
    "repo": REPO, "checkpoint_commit": Path(CKPT_DIR).name,
    "parameters": sum(p.numel() for p in model.parameters()), "n_out_tokens": N_OUT,
    "load_seconds": time.time() - t0,
    "config_sha256": hashlib.sha256(Path(CKPT_DIR, "config.json").read_bytes()).hexdigest(),
}
print(json.dumps(model_info, indent=2), flush=True)
"""),
    code("""def decode(ids):
    value = tok.token_ids_to_floats(ids)
    return float(value[0] if isinstance(value, (list, tuple)) else value)

@torch.inference_mode()
def evaluate(task, inputs, targets, max_length=2048, batch_size=BATCH_SIZE):
    started = time.time(); records = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start:start+batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True,
                  max_length=max_length).to(device)
        draws = []
        for _ in range(NUM_SAMPLES):
            out = model.generate(
                **enc, do_sample=True, top_p=0.95, temperature=1.0,
                min_new_tokens=N_OUT, max_new_tokens=N_OUT,
                pad_token_id=getattr(tok, "pad_token_id", 0), use_cache=True,
            )
            draws.append([decode(row.tolist()) for row in out])
        draws = np.asarray(draws, dtype=float).T
        preds = np.nanmedian(draws, axis=1)
        for j, (text, target, pred) in enumerate(zip(chunk, targets[start:start+len(chunk)], preds)):
            records.append({
                "task": task, "i": start+j,
                "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "target": float(target), "prediction": float(pred),
                **{f"draw_{d}": float(draws[j, d]) for d in range(NUM_SAMPLES)},
            })
        print(f"[{task}] batch {start//batch_size+1}/{math.ceil(len(inputs)/batch_size)} "
              f"({start+len(chunk)}/{len(inputs)}) elapsed={time.time()-started:.1f}s", flush=True)
    frame = pd.DataFrame(records)
    valid = np.isfinite(frame.target) & np.isfinite(frame.prediction)
    rho = float(stats.spearmanr(frame.loc[valid, "target"], frame.loc[valid, "prediction"]).statistic)
    frame.to_csv(RESULTS / f"{task}.csv", index=False)
    result = {"task": task, "n": int(valid.sum()), "spearman": rho,
              "seconds": time.time()-started, "num_samples": NUM_SAMPLES}
    print("RESULT", json.dumps(result), flush=True)
    return result
"""),
    code("""# Claim 3: fetch all 17 CodeNet language buckets in one parquet pass.
CODENET_17 = ["C++", "Python", "Java", "C", "Ruby", "C#", "Rust", "Go", "Haskell",
              "Kotlin", "JavaScript", "PHP", "D", "Scala", "OCaml", "Perl", "Fortran"]
codenet_results = []
if RUN_CODENET_17:
    print("Downloading released Code-Regression parquet (5.6 GB, cached by Colab)...", flush=True)
    parquet = hf_hub_download("akhauriyash/Code-Regression", "data.parquet", repo_type="dataset")
    dataset = ds.dataset(parquet, format="parquet")
    flt = pa.compute.equal(pa.compute.field("space"), "CDSS")
    buckets = {lang: [] for lang in CODENET_17}; scanned = 0
    for batch in dataset.scanner(columns=["input", "target", "metadata"], filter=flt,
                                 batch_size=2048).to_batches():
        for inp, target, metadata in zip(batch.column("input"), batch.column("target"), batch.column("metadata")):
            scanned += 1
            try: meta = literal_eval(metadata.as_py()) if metadata.is_valid else {}
            except Exception: meta = {}
            lang = meta.get("language")
            if lang in buckets and len(buckets[lang]) < ROWS_PER_LANGUAGE and inp.is_valid and target.is_valid:
                buckets[lang].append((f"# CDSS\\n# Language: {lang}\\n{inp.as_py()}", float(target.as_py())))
        filled = sum(len(v) >= ROWS_PER_LANGUAGE for v in buckets.values())
        if scanned % 100000 < 2048: print(f"fetch: scanned={scanned}, filled={filled}/17", flush=True)
        if filled == 17: break
    print("bucket sizes", {k: len(v) for k, v in buckets.items()}, flush=True)
    assert all(len(v) == ROWS_PER_LANGUAGE for v in buckets.values())
    for lang in CODENET_17:
        pairs = buckets[lang]
        codenet_results.append(evaluate(
            "codenet_" + lang.replace("+", "p").replace("#", "sharp").lower(),
            [x for x, _ in pairs], [y for _, y in pairs], max_length=2048,
        ) | {"language": lang})
    avg = float(np.mean([r["spearman"] for r in codenet_results]))
    print(f"CLAIM 3: average Spearman across 17 languages = {avg:.6f} (>0.5 required)", flush=True)
"""),
    code("""# Claim 1 missing metric: server-side filtering avoids downloading the 4.1 GB ONNX parquet.
def fetch_grapharch(space, limit):
    url = "https://datasets-server.huggingface.co/filter"
    where = '"space"=' + repr(space)
    params = {"dataset": "akhauriyash/GraphArch-Regression", "config": "default",
              "split": "train", "where": where, "offset": 0, "length": limit}
    response = requests.get(url, params=params, timeout=600); response.raise_for_status()
    rows = response.json()["rows"]
    assert len(rows) == limit, (space, len(rows), limit)
    return [f"{space}\\n\\n{r['row']['input']}" for r in rows], [float(r["row"]["val_accuracy"]) for r in rows]

onnx_results = []
if RUN_ONNX_ACCURACY:
    for space in ONNX_SPACES:
        inputs, targets = fetch_grapharch(space, ONNX_ROWS_PER_SPACE)
        print(f"ONNX fetch {space}: n={len(inputs)}, target range={min(targets):.3f}..{max(targets):.3f}", flush=True)
        # ONNX strings are much longer than source-code rows; batch 2 is safe on T4.
        onnx_results.append(evaluate(
            "onnx_" + space.lower(), inputs, targets, max_length=4096, batch_size=2
        ) | {"space": space})
    print("CLAIM 1 ONNX accuracy results", json.dumps(onnx_results, indent=2), flush=True)
"""),
    code("""# Optional accepted baselines; disabled by default to save GPU time.
accepted_results = []
if RUN_ALREADY_ACCEPTED_APPS_KBSS:
    if "dataset" not in globals():
        parquet = hf_hub_download("akhauriyash/Code-Regression", "data.parquet", repo_type="dataset")
        dataset = ds.dataset(parquet, format="parquet")
    for space in ["APPS", "KBSS"]:
        flt = pa.compute.equal(pa.compute.field("space"), space); pairs = []
        for batch in dataset.scanner(columns=["input", "target"], filter=flt, batch_size=2048).to_batches():
            for inp, target in zip(batch.column("input"), batch.column("target")):
                if inp.is_valid and target.is_valid: pairs.append((f"{space}\\n{inp.as_py()}", float(target.as_py())))
                if len(pairs) == 512: break
            if len(pairs) == 512: break
        accepted_results.append(evaluate(space.lower(), [x for x, _ in pairs], [y for _, y in pairs]))
"""),
    code("""# Validate, write a self-contained evidence summary, and download one ZIP.
summary = {
    "environment": environment, "model": model_info,
    "configuration": {"rows_per_language": ROWS_PER_LANGUAGE, "onnx_rows_per_space": ONNX_ROWS_PER_SPACE,
                      "num_samples": NUM_SAMPLES, "batch_size": BATCH_SIZE, "seed": SEED},
    "claim_3_codenet": {
        "per_language": codenet_results,
        "average_spearman": float(np.mean([r["spearman"] for r in codenet_results])) if codenet_results else None,
        "threshold": 0.5,
    },
    "claim_1_onnx_accuracy": onnx_results,
    "accepted_optional": accepted_results,
}
if codenet_results:
    assert len(codenet_results) == 17 and all(r["n"] == ROWS_PER_LANGUAGE for r in codenet_results)
    assert summary["claim_3_codenet"]["average_spearman"] > 0.5
if onnx_results:
    assert len(onnx_results) == len(ONNX_SPACES) and all(r["n"] == ONNX_ROWS_PER_SPACE for r in onnx_results)
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\\n")
zip_path = shutil.make_archive("regresslm_evidence_bundle", "zip", RESULTS)
print(json.dumps(summary, indent=2), flush=True)
print("BUNDLE", zip_path, os.path.getsize(zip_path), "bytes", flush=True)
from google.colab import files
files.download(zip_path)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUT)

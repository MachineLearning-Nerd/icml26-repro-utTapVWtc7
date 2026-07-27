#!/usr/bin/env python3
"""Fixed OpenResearch entrypoint; behavior varies only through committed config."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from run_claim4_checkpoint import run as run_claim4_checkpoint
from verify_claim4_audit import verify as verify_claim4_audit
from verify_cumulative import verify as verify_cumulative


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "repro/config/campaign.json"


def cpu_allocation() -> dict:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    return {
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": affinity,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    started = time.monotonic()
    config = json.loads(CONFIG_PATH.read_text())
    if config["compute"]["gpu_allowed"]:
        raise AssertionError("campaign configuration must remain CPU-only")
    if config["phase"] not in {
        "baseline", "claim4_released_audit", "claim4_cpu_calibration"
    }:
        raise AssertionError(f"unsupported campaign phase: {config['phase']}")
    print("CAMPAIGN_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    print("RUNTIME_START " + json.dumps({
        "git_sha": git_sha(),
        "cpu_allocation": cpu_allocation(),
        "estimated_required_cores": config["compute"]["estimated_required_cores"],
        "selected_backend": config["compute"]["backend"],
        "selected_flavor": config["compute"]["flavor"],
    }, sort_keys=True), flush=True)
    with threadpool_limits(limits=config["compute"]["estimated_required_cores"]):
        cumulative = verify_cumulative()
        if config["phase"] == "baseline":
            result = cumulative
        elif config["phase"] == "claim4_released_audit":
            result = verify_claim4_audit()
            result["cumulative_claims_1_to_3"] = cumulative
        else:
            result = run_claim4_checkpoint(config)
            result["claim4_released_audit"] = verify_claim4_audit()
            result["cumulative_claims_1_to_3"] = cumulative
    elapsed = time.monotonic() - started
    final = {
        "status": result["status"],
        "phase": config["phase"],
        "git_sha": git_sha(),
        "runtime_seconds": elapsed,
        "cpu_allocation": cpu_allocation(),
        "evidence": result,
    }
    print("CAMPAIGN_RESULT " + json.dumps(final, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline evaluator-visible and protected-history gate for a Space candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path


LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SECRET = re.compile(
    rb"(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    rb"AKIA[0-9A-Z]{16}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)"
)
CURRENT_CLAIMS = {
    1: "pages/current-claim-1/page.md",
    2: "pages/current-claim-2/page.md",
    3: "pages/current-claim-3/page.md",
    4: "pages/claim-4-kendall-ranking/page.md",
    5: "pages/claim-5-ablation-scaling/page.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protected_manifest(path: Path) -> dict[str, str]:
    output = {}
    for line in path.read_text().splitlines():
        digest, filename = line.split("  ", 1)
        output[filename] = digest
    return output


def changed_paths(candidate: Path) -> list[str]:
    modified = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=candidate,
        text=True,
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=candidate,
        text=True,
    ).splitlines()
    return sorted(set(modified + untracked))


def crawl(candidate: Path) -> tuple[list[str], list[dict]]:
    queue = deque(["README.md"])
    visited = set()
    missing = []
    while queue:
        relative = queue.popleft()
        if relative in visited:
            continue
        target = candidate / relative
        if not target.is_file():
            missing.append({"source": "crawl", "target": relative})
            continue
        visited.add(relative)
        if target.suffix.lower() not in {".md", ".json"}:
            continue
        text = target.read_text(errors="replace")
        if relative == "logbook.json":
            payload = json.loads(text)
            queue.append(payload["root"]["file"])
            queue.extend(row["file"] for row in payload["root"]["children"])
        for raw_link in LINK.findall(text):
            link = raw_link.split("#", 1)[0]
            if not link or "://" in link or link.startswith("mailto:"):
                continue
            resolved = (target.parent / link).resolve()
            try:
                next_relative = str(resolved.relative_to(candidate.resolve()))
            except ValueError:
                missing.append({"source": relative, "target": raw_link})
                continue
            if not resolved.is_file():
                missing.append({"source": relative, "target": raw_link})
                continue
            queue.append(next_relative)
        if relative == "README.md":
            queue.append("logbook.json")
    return sorted(visited), missing


def verify(candidate: Path, protected_manifest: Path) -> dict:
    candidate = candidate.resolve()
    protected = load_protected_manifest(protected_manifest)
    files = {
        str(path.relative_to(candidate))
        for path in candidate.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    missing_old_files = sorted(set(protected) - files)
    historical_page_hash_mismatches = sorted(
        filename for filename, digest in protected.items()
        if filename.startswith("pages/")
        and (candidate / filename).is_file()
        and sha256(candidate / filename) != digest
    )
    allowlist = (candidate / "UPLOAD_ALLOWLIST.txt").read_text().splitlines()
    actual_changes = changed_paths(candidate)
    allowlist_mismatch = {
        "not_changed": sorted(set(allowlist) - set(actual_changes)),
        "not_allowlisted": sorted(set(actual_changes) - set(allowlist)),
    }
    manifest = load_protected_manifest(candidate / "MANIFEST.sha256")
    manifest_mismatches = sorted(
        filename for filename, digest in manifest.items()
        if not (candidate / filename).is_file()
        or sha256(candidate / filename) != digest
    )
    secret_files = sorted(
        filename for filename in allowlist
        if (candidate / filename).is_file()
        and SECRET.search((candidate / filename).read_bytes())
    )
    visited, missing_links = crawl(candidate)
    release = (candidate / "pages/release-report/page.md").read_text()
    current_checks = {}
    for claim, relative in CURRENT_CLAIMS.items():
        text = (candidate / relative).read_text()
        expected = "VERIFIED" if claim <= 3 else "BLOCKED"
        current_checks[str(claim)] = {
            "page": relative,
            "reachable": relative in visited,
            "verdict_visible": expected in text,
            "listed_in_visibility_matrix": (
                f"| {claim} |" in release
                and relative.split("/", 1)[1] in release
            ),
        }
    gates = {
        "claim4_exact_gate_reachable": "code/claim4_exact_gate.py" in visited,
        "claim5_exact_gate_reachable": "code/claim5_exact_gate.py" in visited,
        "claim4_raw_reachable": "evidence/claim4/raw_audit_output.json" in visited,
        "claim5_raw_reachable": "evidence/claim5/raw_audit_output.json" in visited,
        "environment_lock_reachable": "environment/uv.lock" in visited,
    }
    release_checker = subprocess.run(
        [sys.executable, str(candidate / "code/verify_space_release.py")],
        cwd=candidate,
        text=True,
        capture_output=True,
        check=False,
    )
    claim4_gate = subprocess.run(
        [sys.executable, str(candidate / "code/claim4_exact_gate.py")],
        cwd=candidate,
        text=True,
        capture_output=True,
        check=False,
    )
    claim5_gate = subprocess.run(
        [sys.executable, str(candidate / "code/claim5_exact_gate.py")],
        cwd=candidate,
        text=True,
        capture_output=True,
        check=False,
    )
    executable_checks = {
        "release_checker_exit_code": release_checker.returncode,
        "release_checker_passed": (
            release_checker.returncode == 0
            and "SPACE_RELEASE_RESULT" in release_checker.stdout
        ),
        "claim4_exact_gate_exit_code": claim4_gate.returncode,
        "claim4_exact_gate_failed_closed": (
            claim4_gate.returncode != 0
            and "CLAIM4_EXACT_GATE_BLOCKED" in claim4_gate.stdout
        ),
        "claim5_exact_gate_exit_code": claim5_gate.returncode,
        "claim5_exact_gate_failed_closed": (
            claim5_gate.returncode != 0
            and "CLAIM5_EXACT_GATE_BLOCKED" in claim5_gate.stdout
        ),
    }
    failures = []
    if missing_old_files:
        failures.append("protected file set is not a subset")
    if historical_page_hash_mismatches:
        failures.append("historical judged pages changed")
    if any(allowlist_mismatch.values()):
        failures.append("upload allowlist does not equal candidate changes")
    if manifest_mismatches:
        failures.append("candidate manifest mismatch")
    if secret_files:
        failures.append("secret-like content detected")
    if missing_links:
        failures.append("relative link traversal has missing targets")
    if not all(
        row["reachable"] and row["verdict_visible"]
        and row["listed_in_visibility_matrix"]
        for row in current_checks.values()
    ):
        failures.append("current claim visibility incomplete")
    if not all(gates.values()):
        failures.append("code/raw/environment traversal incomplete")
    if not all((
        executable_checks["release_checker_passed"],
        executable_checks["claim4_exact_gate_failed_closed"],
        executable_checks["claim5_exact_gate_failed_closed"],
    )):
        failures.append("standalone evaluator-facing executable checks failed")
    return {
        "status": "PASS" if not failures else "FAIL",
        "review_round": 3,
        "start_entrypoint": "README.md",
        "candidate_base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=candidate, text=True
        ).strip(),
        "visited_files": visited,
        "visited_count": len(visited),
        "missing_relative_links": missing_links,
        "protected_old_file_count": len(protected),
        "protected_missing_files": missing_old_files,
        "historical_page_hash_mismatches": historical_page_hash_mismatches,
        "allowlist_count": len(allowlist),
        "allowlist_mismatch": allowlist_mismatch,
        "manifest_count": len(manifest),
        "manifest_mismatches": manifest_mismatches,
        "secret_like_files": secret_files,
        "current_claim_checks": current_checks,
        "gates": gates,
        "executable_checks": executable_checks,
        "conclusions_not_verifiable": [
            (
                "The full-scale APPS/KBSS row-level draws were not retained "
                "in the judged repository. The candidate exposes the accepted "
                "aggregate JSON and historical n=40 row-level control, and "
                "states this limitation inline."
            )
        ],
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("protected_manifest", type=Path)
    args = parser.parse_args()
    result = verify(args.candidate, args.protected_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

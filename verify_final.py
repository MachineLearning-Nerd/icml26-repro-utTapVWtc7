#!/usr/bin/env python3
"""Fail-closed publication checks for the paper-first repository surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANONICAL_NAME = "MachineLearning-Nerd"
CANONICAL_EMAIL = "MachineLearning-Nerd@users.noreply.github.com"
CANONICAL_REPOSITORY = (
    "https://github.com/MachineLearning-Nerd/"
    "icml26-regression-language-models-code"
)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    manifest = load_json("EVIDENCE_MANIFEST.json")

    for relative in manifest["required_files"]:
        if not (ROOT / relative).exists():
            errors.append(f"missing required file: {relative}")

    claims = load_json("claims.json")
    claim_map = {claim["id"]: claim for claim in claims["claims"]}
    expected_statuses = {
        "C1": "VERIFIED_SCOPED",
        "C2": "VERIFIED_SCOPED",
        "C3": "VERIFIED_SCOPED",
        "C4": "BLOCKED_LOW",
        "C5": "BLOCKED_LOW",
    }
    for claim_id, expected in expected_statuses.items():
        actual = claim_map.get(claim_id, {}).get("status")
        if actual != expected:
            errors.append(f"{claim_id} status is {actual!r}, expected {expected!r}")
        if not claim_map.get(claim_id, {}).get("production_path"):
            errors.append(f"{claim_id} has no production path")
        if not claim_map.get(claim_id, {}).get("evidence"):
            errors.append(f"{claim_id} has no evidence list")

    claim1 = load_json("outputs/claim1_validation.json")
    if nested(claim1, "status") != "PASS":
        errors.append("claim1_validation.json is not PASS")
    if nested(claim1, "approaches_executed") != 10:
        errors.append("Claim 1 does not record exactly 10 executed approaches")

    source_audit = load_json("outputs/claim1_source_audit.json")
    if nested(source_audit, "status") != "PASS":
        errors.append("claim1_source_audit.json is not PASS")

    bundle = load_json("outputs/colab/evidence_bundle_verification.json")
    if nested(bundle, "status") != "PASS":
        errors.append("evidence_bundle_verification.json is not PASS")

    claim4 = load_json(
        ".openresearch/artifacts/claim4_released_audit/four_routes.json"
    )
    claim4_checker = load_json(
        ".openresearch/artifacts/claim4_released_audit/"
        "independent_checker_output.json"
    )
    if nested(claim4, "final_verdict") != "BLOCKED":
        errors.append("Claim 4 four-route verdict is not BLOCKED")
    if nested(claim4_checker, "status") != "PASS":
        errors.append("Claim 4 independent checker is not PASS")
    if nested(claim4_checker, "claim_verdict") != "BLOCKED":
        errors.append("Claim 4 checker does not record a BLOCKED claim")

    claim5 = load_json(
        ".openresearch/artifacts/claim5_ablation_scaling/four_routes.json"
    )
    claim5_checker = load_json(
        ".openresearch/artifacts/claim5_ablation_scaling/"
        "independent_checker_output.json"
    )
    if nested(claim5, "final_verdict") != "BLOCKED":
        errors.append("Claim 5 four-route verdict is not BLOCKED")
    if nested(claim5_checker, "status") != "PASS":
        errors.append("Claim 5 independent checker is not PASS")
    if nested(claim5_checker, "claim_verdict") != "BLOCKED":
        errors.append("Claim 5 checker does not record a BLOCKED claim")

    expected_hashes = manifest.get("artifact_sha256", {})
    for relative, expected_hash in expected_hashes.items():
        if not expected_hash:
            errors.append(f"manifest hash is empty: {relative}")
            continue
        path = ROOT / relative
        if not path.exists():
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            errors.append(
                f"hash mismatch for {relative}: {actual_hash} != {expected_hash}"
            )

    remote = git("config", "--get", "remote.origin.url")
    normalized_remote = remote
    if normalized_remote.startswith("git@github.com:"):
        normalized_remote = "https://github.com/" + normalized_remote.split(
            ":", 1
        )[1]
    if normalized_remote.endswith(".git"):
        normalized_remote = normalized_remote[:-4]
    if normalized_remote != CANONICAL_REPOSITORY:
        errors.append(
            f"origin is {normalized_remote!r}, expected {CANONICAL_REPOSITORY!r}"
        )

    expected_branches = set(manifest["expected_branches"])
    local_branches = set(
        filter(
            None,
            git(
                "for-each-ref",
                "--format=%(refname:short)",
                "refs/heads",
            ).splitlines(),
        )
    )
    remote_branches = set()
    for branch in git(
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/remotes/origin",
    ).splitlines():
        if branch.startswith("origin/") and branch != "origin/HEAD":
            remote_branches.add(branch[len("origin/") :])
    if local_branches != expected_branches and remote_branches != expected_branches:
        errors.append(
            "branch inventory mismatch: "
            f"local={sorted(local_branches)}, "
            f"remote={sorted(remote_branches)}, "
            f"expected={sorted(expected_branches)}"
        )
    all_branch_names = local_branches | remote_branches
    for branch in sorted(all_branch_names):
        if (
            branch == "master"
            or branch.startswith("orx/")
            or branch.startswith("publication/")
        ):
            errors.append(f"legacy branch remains: {branch}")

    commit_rows = git(
        "log",
        "--all",
        "--format=%H%x09%an%x09%ae",
    ).splitlines()
    for row in commit_rows:
        parts = row.split("\t", 2)
        if len(parts) != 3:
            errors.append(f"unreadable commit identity row: {row}")
            continue
        commit, author_name, author_email = parts
        if author_name != CANONICAL_NAME or author_email != CANONICAL_EMAIL:
            errors.append(
                f"non-canonical author at {commit}: "
                f"{author_name} <{author_email}>"
            )
    messages = git("log", "--all", "--format=%B")
    if "Co-authored-by:" in messages or "Co-Authored-By:" in messages:
        errors.append("co-author trailer found in reachable history")

    if errors:
        print("VERIFY_FINAL_FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "VERIFY_FINAL_PASS: "
        f"{len(manifest['required_files'])} required files, "
        f"{len(expected_branches)} branches, 5 claim contracts"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

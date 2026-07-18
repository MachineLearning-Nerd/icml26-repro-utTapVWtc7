#!/usr/bin/env python3
"""Fail-closed publication gate for the exactly-ten-route Claim-1 repair."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CLAIM1_PUBLISH_GATE_FAIL: {message}")


def main() -> None:
    validation_path = ROOT / "outputs/claim1_validation.json"
    require(validation_path.exists(), "validation JSON missing")
    result = json.loads(validation_path.read_text())
    require(result["approaches_executed"] == 10, "approach count is not exactly ten")
    require(
        [row["number"] for row in result["approaches"]] == list(range(1, 11)),
        "routes are not numbered 1 through 10 exactly once",
    )
    require(result["accuracy"]["rows"] == 1536, "expected 3 x 512 accuracy rows")
    require(result["accuracy"]["raw_draws"] == 12_288, "expected eight draws per accuracy row")
    require(
        {row["space"] for row in result["accuracy"]["per_space"]}
        == {"NASBench101", "ENAS", "NASNet"},
        "author-card accuracy-space coverage is incomplete",
    )

    ledger = (ROOT / "CLAIM1_APPROACH_LEDGER.md").read_text()
    require(all(f"| {index} |" in ledger for index in range(1, 11)), "ledger routes missing")
    require("| 11 |" not in ledger, "more than ten routes listed")
    require((ROOT / "CLAIM1_SOURCE_AUDIT.md").exists(), "source/scope audit missing")
    source_audit = json.loads((ROOT / "outputs/claim1_source_audit.json").read_text())
    require(source_audit["status"] == "PASS", "machine-readable source audit failed")
    require(source_audit["dataset"]["local_matches_hub"], "full parquet does not match Hub LFS")
    require((ROOT / "outputs/claim1_accuracy/full_n512.csv").exists(), "raw accuracy CSV missing")

    pages = sorted((ROOT / ".trackio/logbook/pages").glob("*/page.md"))
    page_text = "\n".join(page.read_text() for page in pages)
    require(page_text.count('"pinned": true') == 1, "expected exactly one pinned report cell")
    require("/home/" not in page_text, "absolute local path leaked into report")
    require("exactly 10" in page_text.lower(), "exact-ten disclosure missing")
    require("NASBench101" in page_text and "ENAS" in page_text and "NASNet" in page_text,
            "three accuracy spaces not disclosed")
    require("512" in page_text and "12,288" in page_text, "paper-scale row/draw counts missing")
    require("input-shuffle" in page_text.lower(), "falsification control missing")

    metadata = json.loads((ROOT / ".trackio/metadata.json").read_text())
    require(metadata["tags"] == ["icml2026-repro", "paper-utTapVWtc7"], "Space tags wrong")
    require(metadata["private"] is False and metadata["autosync"] is False,
            "publication metadata is not fail-closed")
    require(not metadata.get("local_path_artifacts"), "absolute local artifact metadata remains")
    print(
        "CLAIM1_PUBLISH_GATE_PASS approaches=10 accuracy_rows=1536 "
        "accuracy_draws=12288 spaces=3 pins=1"
    )


if __name__ == "__main__":
    main()

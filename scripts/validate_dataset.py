#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REQUIRED = {"id", "domain", "doc_type", "title", "citation", "year", "source_url", "tags", "snippet", "retrieval_priority"}
ALLOWED_DOMAIN = {"criminal_law", "family_law"}
ALLOWED_TYPE = {"statute", "case_law", "commentary"}
ALLOWED_PRIORITY = {"high", "medium", "low"}


def main(path_str: str) -> int:
    path = Path(path_str)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    errors = []
    ids = set()
    total = 0

    with path.open("r", encoding="utf-8") as fh:
        for i, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {i}: invalid json ({exc})")
                continue

            missing = REQUIRED - obj.keys()
            if missing:
                errors.append(f"line {i}: missing keys {sorted(missing)}")

            row_id = obj.get("id")
            if row_id in ids:
                errors.append(f"line {i}: duplicate id '{row_id}'")
            ids.add(row_id)

            if obj.get("domain") not in ALLOWED_DOMAIN:
                errors.append(f"line {i}: invalid domain '{obj.get('domain')}'")
            if obj.get("doc_type") not in ALLOWED_TYPE:
                errors.append(f"line {i}: invalid doc_type '{obj.get('doc_type')}'")
            if obj.get("retrieval_priority") not in ALLOWED_PRIORITY:
                errors.append(f"line {i}: invalid retrieval_priority '{obj.get('retrieval_priority')}'")
            if not isinstance(obj.get("tags"), list) or not obj.get("tags"):
                errors.append(f"line {i}: tags must be a non-empty list")

    if errors:
        print("Validation FAILED")
        for err in errors:
            print(f" - {err}")
        return 1

    print(f"Validation PASSED: {total} records")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/validate_dataset.py <path-to-jsonl>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))

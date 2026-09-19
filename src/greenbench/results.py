from __future__ import annotations

import json
from pathlib import Path

from .provenance import RunRecord


def write_immutable(record: RunRecord, root: Path) -> Path:
    folder = root / record.classification.value.lower()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{record.run_id}.json"
    if path.exists():
        raise FileExistsError(f"Raw run is immutable: {path}")
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def validate_directory(root: Path) -> tuple[int, list[str]]:
    errors = []
    count = 0
    for path in root.rglob("*.json") if root.exists() else []:
        try:
            RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
            count += 1
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return count, errors


def aggregate(root: Path, output: Path) -> int:
    rows = []
    for path in root.rglob("*.json") if root.exists() else []:
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return len(rows)

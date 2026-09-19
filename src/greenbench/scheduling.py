from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import content_hash


def create_schedule(seeds: list[int], datasets: list[str], seed: int = 2026) -> dict:
    rng = np.random.default_rng(seed)
    entries = []
    order = 0
    for dataset in datasets:
        for paired_seed in seeds:
            frameworks = ["pytorch", "tensorflow"]
            rng.shuffle(frameworks)
            for framework in frameworks:
                order += 1
                entries.append(
                    {
                        "order": order,
                        "dataset": dataset,
                        "seed": paired_seed,
                        "framework": framework,
                        "status": "pending",
                    }
                )
    schedule = {"schedule_seed": seed, "entries": entries}
    schedule["hash"] = content_hash(schedule)
    return schedule


def save_schedule(schedule: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Schedule already exists: {path}")
    path.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")

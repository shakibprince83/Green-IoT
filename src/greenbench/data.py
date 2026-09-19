from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def stratified_split(labels: np.ndarray, validation_size: int, seed: int) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if not 0 < validation_size < labels.size:
        raise ValueError("validation_size must be between zero and the training size")
    rng = np.random.default_rng(seed)
    val_parts: list[np.ndarray] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        count = int(round(validation_size * len(idx) / len(labels)))
        val_parts.append(idx[:count])
    validation = np.sort(np.concatenate(val_parts))
    if len(validation) != validation_size:
        pool = np.setdiff1d(np.arange(labels.size), validation)
        delta = validation_size - len(validation)
        validation = (
            np.sort(np.concatenate([validation, pool[:delta]]))
            if delta > 0
            else validation[:validation_size]
        )
    train = np.setdiff1d(np.arange(labels.size), validation)
    return {"train": train.astype(np.int64), "validation": validation.astype(np.int64)}


def split_hash(parts: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(parts):
        digest.update(name.encode())
        digest.update(np.asarray(parts[name], dtype=np.int64).tobytes())
    return digest.hexdigest()


def save_split(path: Path, parts: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = split_hash(parts)
    np.savez_compressed(path, **parts, checksum=np.array(checksum))
    return checksum


def load_split(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as raw:
        expected = str(raw["checksum"])
        parts = {name: raw[name] for name in raw.files if name != "checksum"}
    actual = split_hash(parts)
    if actual != expected:
        raise ValueError(f"Split artifact corruption: expected {expected}, got {actual}")
    return parts, actual


def epoch_orders(size: int, epochs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(size) for _ in range(epochs)])

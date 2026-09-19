from __future__ import annotations

import hashlib

import numpy as np

SHAPES = {
    "conv1.weight": (32, 1, 3, 3),
    "conv1.bias": (32,),
    "conv2.weight": (64, 32, 3, 3),
    "conv2.bias": (64,),
    "dense1.weight": (128, 9216),
    "dense1.bias": (128,),
    "dense2.weight": (10, 128),
    "dense2.bias": (10,),
}


def parameter_count() -> int:
    return sum(int(np.prod(shape)) for shape in SHAPES.values())


def canonical_weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[str, np.ndarray] = {}
    for name, shape in SHAPES.items():
        if name.endswith("bias"):
            value = np.zeros(shape, dtype=np.float32)
        else:
            fan_in = int(np.prod(shape[1:]))
            bound = np.sqrt(6.0 / fan_in)
            value = rng.uniform(-bound, bound, shape).astype(np.float32)
        result[name] = value
    return result


def initialization_hash(weights: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(weights):
        digest.update(name.encode())
        digest.update(weights[name].astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def to_tensorflow(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "conv1.kernel": weights["conv1.weight"].transpose(2, 3, 1, 0),
        "conv1.bias": weights["conv1.bias"],
        "conv2.kernel": weights["conv2.weight"].transpose(2, 3, 1, 0),
        "conv2.bias": weights["conv2.bias"],
        "dense1.kernel": weights["dense1.weight"].T,
        "dense1.bias": weights["dense1.bias"],
        "dense2.kernel": weights["dense2.weight"].T,
        "dense2.bias": weights["dense2.bias"],
    }


def from_tensorflow(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "conv1.weight": weights["conv1.kernel"].transpose(3, 2, 0, 1),
        "conv1.bias": weights["conv1.bias"],
        "conv2.weight": weights["conv2.kernel"].transpose(3, 2, 0, 1),
        "conv2.bias": weights["conv2.bias"],
        "dense1.weight": weights["dense1.kernel"].T,
        "dense1.bias": weights["dense1.bias"],
        "dense2.weight": weights["dense2.kernel"].T,
        "dense2.bias": weights["dense2.bias"],
    }

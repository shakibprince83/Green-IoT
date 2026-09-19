from datetime import datetime, timezone

import numpy as np
import pytest

from greenbench.data import epoch_orders, load_split, save_split, split_hash, stratified_split
from greenbench.green_metrics import green_efficiency, pareto_front, symmetric_difference
from greenbench.initialization import (
    canonical_weights,
    from_tensorflow,
    parameter_count,
    to_tensorflow,
)
from greenbench.provenance import RunRecord
from greenbench.scheduling import create_schedule
from greenbench.statistics import corrections, require_complete_pairs


def test_split_hash_stable_and_corruption_detected(tmp_path):
    parts = stratified_split(np.repeat(np.arange(10), 10), 20, 7)
    assert split_hash(parts) == split_hash(parts)
    path = tmp_path / "split.npz"
    save_split(path, parts)
    loaded, digest = load_split(path)
    assert digest == split_hash(loaded)
    with np.load(path) as raw:
        broken = {k: raw[k] for k in raw.files}
    broken["train"][0] += 1
    np.savez(path, **broken)
    with pytest.raises(ValueError):
        load_split(path)


def test_parameters_and_initialization_roundtrip():
    assert parameter_count() == 1_199_882
    original = canonical_weights(11)
    restored = from_tensorflow(to_tensorflow(original))
    assert all(np.array_equal(original[k], restored[k]) for k in original)


def test_orders_and_schedule_deterministic():
    assert np.array_equal(epoch_orders(20, 3, 8), epoch_orders(20, 3, 8))
    assert create_schedule([11, 22], ["mnist"]) == create_schedule([11, 22], ["mnist"])


def test_green_metrics():
    assert symmetric_difference(10, 10) == 0
    rows = [
        {"macro_f1": 0.9, "throughput": 2, "energy": 1, "train_time": 1, "peak_memory": 1},
        {"macro_f1": 0.8, "throughput": 1, "energy": 2, "train_time": 2, "peak_memory": 2},
    ]
    score = green_efficiency(
        rows,
        {"macro_f1": 0.2, "throughput": 0.2, "energy": 0.2, "train_time": 0.2, "peak_memory": 0.2},
    )
    assert score[0] > score[1]
    assert pareto_front(np.array([[0.9, 10], [0.8, 12], [0.85, 8]]), (True, False)).tolist() == [
        True,
        False,
        True,
    ]


def test_corrections_and_pairing():
    adjusted = corrections([0.01, 0.04])
    assert adjusted["bonferroni"].tolist() == [0.02, 0.08]
    rows = [{"dataset": "m", "architecture": "c", "seed": 1, "framework": "pytorch"}]
    with pytest.raises(ValueError):
        require_complete_pairs(rows)


def base_record(**updates):
    values = dict(
        run_id="r",
        experiment_id="e",
        experiment_family="core_framework",
        classification="SMOKE",
        protocol_version="v",
        protocol_hash="a",
        config_hash="b",
        timestamp_start=datetime.now(timezone.utc),
        machine_id="m",
        hardware_hash="h",
        os="linux",
        python_version="3",
        git_commit="g",
        git_dirty=False,
        framework="fixture",
        model="matched_cnn",
        architecture_hash="a",
        initialization_hash="i",
        dataset="fixture",
        dataset_source="test fixture",
        split_hash="s",
        seed=1,
        execution_order=1,
        device="cpu",
        precision="fp32",
        measurement_tier=3,
        measurement_backend="clock",
        measurement_method="wall",
        measurement_scope="training",
        batch_size=2,
        epochs=1,
        optimizer="adam",
        optimizer_hyperparameters={},
        training_samples=2,
        validation_samples=1,
        test_samples=1,
        run_status="success",
    )
    values.update(updates)
    return values


def test_schema_blocks_proxy_energy_and_fake_official():
    RunRecord(**base_record())
    with pytest.raises(ValueError):
        RunRecord(**base_record(training_energy_j=4))
    with pytest.raises(ValueError):
        RunRecord(**base_record(classification="OFFICIAL"))

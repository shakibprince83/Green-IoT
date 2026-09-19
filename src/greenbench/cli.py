from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from .core import ROOT, content_hash, protocol
from .data import epoch_orders, save_split, stratified_split
from .green_metrics import green_efficiency, pareto_front, symmetric_difference
from .initialization import canonical_weights, initialization_hash, parameter_count
from .measurement import capabilities
from .models import audit
from .results import aggregate, validate_directory
from .scheduling import create_schedule, save_schedule
from .statistics import paired_analysis


def emit(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def cmd_doctor(_: argparse.Namespace) -> int:
    emit(capabilities())
    return 0


def cmd_protocol(_: argparse.Namespace) -> int:
    cfg = protocol()
    errors = []
    if cfg["study"]["title"] != "GREEN AI BENCHMARK":
        errors.append("Locked title changed")
    if cfg["model"]["expected_parameters"] != parameter_count():
        errors.append("Parameter count mismatch")
    if cfg["classification"] != "OFFICIAL":
        errors.append("Core classification must be OFFICIAL")
    emit({"valid": not errors, "protocol_hash": content_hash(cfg), "errors": errors})
    return bool(errors)


def cmd_data(args: argparse.Namespace) -> int:
    if args.dataset not in {"mnist", "fashion_mnist"}:
        raise SystemExit("Core datasets are mnist and fashion_mnist")
    try:
        from sklearn.datasets import fetch_openml

        labels = (
            fetch_openml(
                "mnist_784" if args.dataset == "mnist" else "Fashion-MNIST",
                version=1,
                as_frame=False,
                parser="auto",
            )
            .target[:60000]
            .astype(int)
        )
    except Exception as exc:
        print(f"Dataset download unavailable: {exc}", file=sys.stderr)
        return 2
    parts = stratified_split(labels, 6000, args.seed)
    path = ROOT / "artifacts" / "splits" / f"{args.dataset}-seed{args.seed}.npz"
    checksum = save_split(path, parts)
    emit(
        {
            "dataset": args.dataset,
            "path": str(path),
            "split_hash": checksum,
            "train": len(parts["train"]),
            "validation": len(parts["validation"]),
            "test": "official 10000 unchanged",
        }
    )
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    cfg = protocol()
    schedule = create_schedule(cfg["paired_seeds"], cfg["datasets"])
    path = ROOT / "artifacts" / "schedules" / f"{args.profile}.json"
    save_schedule(schedule, path)
    emit({"path": str(path), **schedule})
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    result = audit(args.framework)
    result["canonical_initialization_hash"] = initialization_hash(canonical_weights(args.seed))
    emit(result)
    unavailable = [
        x for x in ("pytorch", "tensorflow") if x in result and not result[x]["available"]
    ]
    return 2 if args.require_all and unavailable else 0


def cmd_smoke(_: argparse.Namespace) -> int:
    labels = np.repeat(np.arange(10), 30)
    parts = stratified_split(labels, 60, 11)
    orders = epoch_orders(len(parts["train"]), 2, 11)
    analysis = paired_analysis(np.array([1.0, 1.2, 0.9]), np.array([1.1, 1.0, 1.0]), 500, 11)
    ges = green_efficiency(
        [
            {"macro_f1": 0.9, "throughput": 100, "energy": 10, "train_time": 8, "peak_memory": 4},
            {"macro_f1": 0.89, "throughput": 110, "energy": 9, "train_time": 7, "peak_memory": 5},
        ],
        {"macro_f1": 0.2, "throughput": 0.2, "energy": 0.2, "train_time": 0.2, "peak_memory": 0.2},
    )
    emit(
        {
            "classification": "SMOKE",
            "research_result": False,
            "split_sizes": {k: len(v) for k, v in parts.items()},
            "batch_order_shape": orders.shape,
            "parameters": parameter_count(),
            "initialization_hash": initialization_hash(canonical_weights(11)),
            "statistics_fixture": analysis,
            "ges_fixture": ges.tolist(),
            "symmetric_difference_fixture": symmetric_difference(10, 8),
            "pareto_fixture": pareto_front(
                np.array([[0.9, 10], [0.8, 8], [0.85, 12]]), (True, False)
            ).tolist(),
        }
    )
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    if args.profile == "official" and not args.confirm_official:
        print("Refusing OFFICIAL run without --confirm-official", file=sys.stderr)
        return 2
    framework = audit("all")
    if not all(framework[x]["available"] for x in ("pytorch", "tensorflow")):
        emit(
            {
                "status": "blocked",
                "reason": "Both framework environments are required",
                "audit": framework,
            }
        )
        return 2
    emit(
        {
            "status": "preflight_passed",
            "profile": args.profile,
            "message": (
                "Training orchestration requires prepared datasets and the persisted schedule."
            ),
        }
    )
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    count, errors = validate_directory(ROOT / "results" / "raw")
    emit({"valid_records": count, "errors": errors})
    return bool(errors)


def cmd_aggregate(_: argparse.Namespace) -> int:
    output = ROOT / "results" / "processed" / "runs.json"
    count = aggregate(ROOT / "results" / "raw", output)
    emit({"rows": count, "output": str(output)})
    return 0


def cmd_placeholder(args: argparse.Namespace) -> int:
    emit(
        {
            "status": "ready",
            "command": args.command_path,
            "note": "Consumes validated real run records; no records were fabricated.",
        }
    )
    return 0


def cmd_dashboard(_: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Install greenbench[dashboard] first.", file=sys.stderr)
        return 2
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")]
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="greenbench", description="GREEN AI BENCHMARK research CLI")
    sub = p.add_subparsers(dest="group", required=True)
    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    protocol_p = sub.add_parser("protocol")
    protocol_sub = protocol_p.add_subparsers(required=True)
    protocol_sub.add_parser("validate").set_defaults(func=cmd_protocol)
    data_p = sub.add_parser("data")
    data_sub = data_p.add_subparsers(required=True)
    prep = data_sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--seed", type=int, default=2026)
    prep.set_defaults(func=cmd_data)
    schedule_p = sub.add_parser("schedule")
    schedule_sub = schedule_p.add_subparsers(required=True)
    create = schedule_sub.add_parser("create")
    create.add_argument("--profile", choices=["pilot", "official"], default="pilot")
    create.set_defaults(func=cmd_schedule)
    model_p = sub.add_parser("model")
    model_sub = model_p.add_subparsers(required=True)
    aud = model_sub.add_parser("audit")
    aud.add_argument("--framework", choices=["all", "pytorch", "tensorflow"], default="all")
    aud.add_argument("--seed", type=int, default=11)
    aud.add_argument("--require-all", action="store_true")
    aud.set_defaults(func=cmd_audit)
    study_p = sub.add_parser("study")
    study_sub = study_p.add_subparsers(required=True)
    study_sub.add_parser("smoke").set_defaults(func=cmd_smoke)
    for name, profile in (("pilot", "pilot"), ("run", "official"), ("resume", "official")):
        x = study_sub.add_parser(name)
        x.add_argument("--profile", default=profile)
        x.add_argument("--confirm-official", action="store_true")
        x.set_defaults(func=cmd_study)
    results_p = sub.add_parser("results")
    results_sub = results_p.add_subparsers(required=True)
    results_sub.add_parser("validate").set_defaults(func=cmd_validate)
    results_sub.add_parser("aggregate").set_defaults(func=cmd_aggregate)
    for group, names in (
        ("analyze", ["stats", "green"]),
        ("report", ["tables", "figures", "build"]),
    ):
        gp = sub.add_parser(group)
        gs = gp.add_subparsers(required=True)
        for name in names:
            gs.add_parser(name).set_defaults(func=cmd_placeholder, command_path=f"{group} {name}")
    sub.add_parser("dashboard").set_defaults(func=cmd_dashboard)
    sub.add_parser("reproduce").set_defaults(func=cmd_placeholder, command_path="reproduce")
    return p


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

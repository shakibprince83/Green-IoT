from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys


def capabilities() -> dict:
    def installed(name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    gpu = shutil.which("nvidia-smi") is not None
    rapl = __import__("pathlib").Path("/sys/class/powercap").exists()
    tier = 1 if gpu or rapl else (2 if installed("codecarbon") or installed("carbontracker") else 3)
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
        "ram_gib": _ram(),
        "gpu": _nvidia(),
        "cuda_tool": shutil.which("nvcc"),
        "pytorch": _version("torch"),
        "tensorflow": _version("tensorflow"),
        "nvml": installed("pynvml"),
        "rapl": rapl,
        "pyjoules": installed("pyJoules"),
        "pyrapl": installed("pyRAPL"),
        "codecarbon": installed("codecarbon"),
        "carbontracker": installed("carbontracker"),
        "psutil": installed("psutil"),
        "supported_tier": tier,
        "limitation": "Tier 3 resource proxies only; no energy claims permitted"
        if tier == 3
        else None,
    }


def _version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(module.__version__)
    except ImportError:
        return None


def _ram() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 2**30, 2)
    except ImportError:
        return None


def _nvidia() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None

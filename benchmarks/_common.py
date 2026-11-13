"""Shared benchmark helpers: environment metadata, CUDA timing, JSON output."""

import datetime as _dt
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def cpu_name() -> str:
    """Human-readable CPU model (best effort)."""
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def env_info() -> dict:
    """GPU name and library versions recorded with every result file."""
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    cuda = torch.cuda.is_available()
    return {
        "date": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "gpu_total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda else None,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton_version,
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def cuda_time(fn: Callable[[], object], warmup: int, iters: int) -> dict:
    """Median / std wall time (ms) of `fn` over `iters` runs after `warmup` runs, synchronizing each run."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "iters": iters,
        "warmup": warmup,
    }


def save_results(name: str, config: dict, results: list[dict], summary: dict | None = None) -> Path:
    """Write {env, config, summary, results} to benchmarks/results/<name>.json and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = {"env": env_info(), "config": config, "summary": summary or {}, "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

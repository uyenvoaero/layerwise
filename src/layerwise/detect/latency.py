"""End-to-end per-layer latency against the recoat-cycle budget.

Measured from a decoded gray image in memory: tiling, model forward, stitching and defect-map
rasterization. JPEG decoding is excluded (and stated in every record).
"""

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from layerwise.detect import recoat, representation
from layerwise.detect.config import DetectConfig
from layerwise.detect.data import load_fold_plan, pb_images, read_gray
from layerwise.detect.jobs import run_root
from layerwise.detect.models.base import Detection, Detector, predict_image
from layerwise.detect.predict import load_run
from layerwise.detect.protocol import DetectorFactory
from layerwise.errors import ConfigError
from layerwise.manifest import write_json

EXCLUDES = ["jpeg_decode"]
SORT_KEY = ("model", "hardware", "runtime")
IDENTITY = ("model", "hardware", "runtime", "batch_size", "input_resolution")


def summarize(timings_ms: list[float], warmup: int) -> dict[str, float]:
    """p50 and p95 of the timed runs (the first `warmup` iterations are dropped)."""
    timed = np.asarray(timings_ms[warmup:], dtype=float)
    if timed.size == 0:
        raise ValueError("no timed runs after warm-up")
    return {
        "runs": int(timed.size),
        "p50_ms": float(np.percentile(timed, 50)),
        "p95_ms": float(np.percentile(timed, 95)),
    }


def build_record(
    *,
    model: str,
    hardware: str,
    device_name: str | None,
    runtime: str,
    precision: str,
    batch_size: int,
    tile_batch_size: int,
    input_resolution: list[int],
    tile_size: int,
    timings_ms: list[float],
    warmup: int,
    budget_ms: float,
) -> dict[str, Any]:
    stats = summarize(timings_ms, warmup)
    within = stats["p95_ms"] <= budget_ms
    return {
        "model": model,
        "hardware": hardware,
        "device_name": device_name,
        "runtime": runtime,
        "precision": precision,
        "batch_size": batch_size,
        "tile_batch_size": tile_batch_size,
        "input_resolution": list(input_resolution),
        "tile_size": tile_size,
        "runs": stats["runs"],
        "warmup": warmup,
        "p50_ms": stats["p50_ms"],
        "p95_ms": stats["p95_ms"],
        "budget_ms": budget_ms,
        "within_budget": within,
        "eligible_default": within,
        "excludes": list(EXCLUDES),
    }


def merge_record(path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Add a record to latency.json; the same setup is replaced, other hardware is kept."""
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    identity = tuple(json.dumps(record[k]) for k in IDENTITY)
    records = [r for r in records if tuple(json.dumps(r[k]) for k in IDENTITY) != identity]
    records.append(record)
    records.sort(key=lambda r: tuple(r[k] for k in SORT_KEY))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, records)
    return records


def measure(
    detector: Detector,
    images: list[np.ndarray],
    params,
    threshold: float,
    warmup: int,
    runs: int,
) -> list[float]:
    """Wall time in ms of `warmup + runs` end-to-end layer passes, cycling over `images`."""
    cuda = detector.info().get("device") == "cuda"
    timings = []
    for i in range(warmup + runs):
        image = images[i % len(images)]
        start = time.perf_counter()
        boxes, scores = predict_image(image, detector, params)
        layer = representation.LayerOutput(
            "latency", "latency", 0, False, image.shape[1], image.shape[0],
            [Detection(tuple(map(float, b)), float(s)) for b, s in zip(boxes, scores, strict=True)],
        )  # fmt: skip
        rows = representation.detection_rows(layer, threshold)
        representation.rasterize(rows, layer.width, layer.height)
        if cuda:
            import torch

            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


@dataclass
class BenchmarkResult:
    record: dict[str, Any]
    recoat: dict[str, Any]


def run_benchmark(
    cfg: DetectConfig, run: str, factory: DetectorFactory, runs: int | None = None
) -> BenchmarkResult:
    """Measure the final model of a run on its held-out layers; always writes both reports."""
    model, k, selection, final = load_run(cfg, run)
    fold = next((f for f in load_fold_plan(cfg).outer_folds if f.outer_fold == k), None)
    if fold is None:
        raise ConfigError(f"outer fold {k} is not in the fold table")
    candidate = next(c for c in cfg.candidates(model) if c.id == selection["winner"])
    detector = factory(model, candidate)
    weights = run_root(cfg, model, k) / "final" / final["weights"] if final["weights"] else None
    detector.load(weights)
    params = cfg.tiling_for(candidate)

    settings = cfg.latency
    warmup = int(settings["warmup"])
    runs = int(runs if runs is not None else settings["runs"])
    images = [read_gray(i.path) for i in pb_images(cfg, [fold.test_build])[: warmup + runs]]
    if not images:
        raise ConfigError(f"no PB images found for build {fold.test_build!r}")
    timings = measure(detector, images, params, selection["threshold"], warmup, runs)

    cycle = recoat.write_recoat_cycle(cfg)
    info = detector.info()
    hardware = "cuda" if info.get("device") == "cuda" else "cpu"
    device_name = info.get("device_name") or (platform.processor() or platform.machine())
    record = build_record(
        model=model,
        hardware=hardware,
        device_name=device_name,
        runtime=info.get("runtime", "pytorch"),
        precision=info.get("precision", "fp32"),
        batch_size=1,
        tile_batch_size=int(info.get("tile_batch_size", 1)),
        input_resolution=[images[0].shape[1], images[0].shape[0]],
        tile_size=params.tile_size,
        timings_ms=timings,
        warmup=warmup,
        budget_ms=cycle["budget_s"] * 1000.0,
    )
    merge_record(cfg.reports_dir / "latency.json", record)
    return BenchmarkResult(record, cycle)

import json

import numpy as np
import pytest

from layerwise.detect import latency

BASE = {
    "model": "yolo",
    "hardware": "cpu",
    "device_name": "Test CPU",
    "runtime": "pytorch",
    "precision": "fp32",
    "batch_size": 1,
    "tile_batch_size": 8,
    "input_resolution": [1280, 1024],
    "tile_size": 640,
}


def record(timings, budget_ms, warmup=2, **overrides):
    return latency.build_record(
        **{**BASE, **overrides}, timings_ms=timings, warmup=warmup, budget_ms=budget_ms
    )


def test_p50_and_p95_are_computed_after_warmup_is_dropped():
    timings = [999.0, 999.0, *[float(v) for v in range(1, 101)]]
    summary = latency.summarize(timings, warmup=2)
    assert summary["runs"] == 100
    assert summary["p50_ms"] == pytest.approx(np.percentile(range(1, 101), 50))
    assert summary["p95_ms"] == pytest.approx(95.05)


def test_within_budget_and_eligible_default_are_false_when_p95_exceeds_the_budget():
    timings = [1.0, 1.0, *[10.0] * 10, *[500.0] * 10]
    over = record(timings, budget_ms=100.0)
    assert over["p95_ms"] > 100.0 and over["within_budget"] is False
    assert over["eligible_default"] is False
    ok = record(timings, budget_ms=1000.0)
    assert ok["within_budget"] is True and ok["eligible_default"] is True


def test_record_has_the_contract_fields():
    rec = record([1.0, 1.0, 2.0, 3.0], budget_ms=5000.0)
    assert set(rec) == {
        "model", "hardware", "device_name", "runtime", "precision", "batch_size",
        "tile_batch_size", "input_resolution", "tile_size", "runs", "warmup", "p50_ms",
        "p95_ms", "budget_ms", "within_budget", "eligible_default", "excludes",
    }  # fmt: skip
    assert rec["excludes"] == ["jpeg_decode"]
    assert rec["runs"] == 2 and rec["warmup"] == 2
    assert rec["input_resolution"] == [1280, 1024]


def test_cpu_and_cuda_records_merge_into_one_sorted_file(tmp_path):
    path = tmp_path / "latency.json"
    latency.merge_record(path, record([1.0, 1.0, 5.0, 6.0], 100.0, model="yolo", hardware="cuda"))
    latency.merge_record(path, record([1.0, 1.0, 9.0, 9.5], 100.0, model="yolo", hardware="cpu"))
    latency.merge_record(path, record([1.0, 1.0, 2.0, 2.5], 100.0, model="contrast"))
    records = json.loads(path.read_text(encoding="utf-8"))
    assert [(r["model"], r["hardware"], r["runtime"]) for r in records] == [
        ("contrast", "cpu", "pytorch"),
        ("yolo", "cpu", "pytorch"),
        ("yolo", "cuda", "pytorch"),
    ]


def test_a_new_measurement_replaces_the_old_one_for_the_same_setup(tmp_path):
    path = tmp_path / "latency.json"
    latency.merge_record(path, record([1.0, 1.0, 9.0, 9.0], 100.0))
    latency.merge_record(path, record([1.0, 1.0, 3.0, 3.0], 100.0))
    records = json.loads(path.read_text(encoding="utf-8"))
    assert len(records) == 1 and records[0]["p50_ms"] == pytest.approx(3.0)

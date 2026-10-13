from datetime import datetime, timedelta

import pytest

from layerwise.detect import recoat
from layerwise.ingest.tables import IMAGES, write_table


def row(build, layer, stamp, modality="PB"):
    return {
        "image_id": f"{build}_{layer:05d}_{stamp}",
        "dataset": "aalto",
        "build_id": build,
        "modality": modality,
        "layer_index": layer,
        "timestamp": stamp,
        "z_mm": None,
        "path": "x.jpg",
        "width": 8,
        "height": 8,
        "mode": "L",
        "size_flag": False,
        "label_status": "labeled",
        "use_for_detection": True,
    }


def stamps(start, gaps, fractional=True):
    """Timestamps like the ingest table: 20240318T133538.870000 (or without the fraction)."""
    moment = datetime(2024, 3, 18, 13, 0, 0)
    out = [moment]
    for gap in gaps:
        moment = moment + timedelta(seconds=gap)
        out.append(moment)
    fmt = "%Y%m%dT%H%M%S.%f" if fractional else "%Y%m%dT%H%M%S"
    return [m.strftime(fmt) for m in out]


@pytest.fixture
def processed(tmp_path):
    def write(rows):
        write_table(tmp_path / "images.parquet", rows, IMAGES)
        return tmp_path

    return write


def test_median_gap_per_build_counts_only_consecutive_layers(processed):
    # layers 1..6 with gaps 20, 22, 24, 600 (a pause), 21; layer 8 follows layer 6 (step 2)
    layers = [1, 2, 3, 4, 5, 6, 8]
    times = stamps(0, [20, 22, 24, 600, 21, 45])
    rows = [row("B1", n, t) for n, t in zip(layers, times, strict=True)]
    result = recoat.recoat_cycle(processed(rows), budget_fraction=0.5)
    entry = result["per_build"]["B1"]
    assert entry["median_gap_s"] == pytest.approx(22.0)  # the 600 s pause does not move it
    assert entry["gaps_used"] == 5  # the layer 6 -> 8 gap is excluded


def test_timestamps_without_fractional_seconds_are_handled(processed):
    times = stamps(0, [10, 12, 14], fractional=False)
    rows = [row("B1", n, t) for n, t in zip([1, 2, 3, 4], times, strict=True)]
    result = recoat.recoat_cycle(processed(rows), budget_fraction=0.5)
    assert result["per_build"]["B1"]["median_gap_s"] == pytest.approx(12.0)
    assert result["per_build"]["B1"]["gaps_used"] == 3


def test_short_fractions_are_padded_to_microseconds(processed):
    rows = [row("B1", 1, "20240318T130000.5"), row("B1", 2, "20240318T130020.750")]
    result = recoat.recoat_cycle(processed(rows), budget_fraction=0.5)
    assert result["per_build"]["B1"]["median_gap_s"] == pytest.approx(20.25)


def test_budget_uses_the_shortest_median_times_the_fraction(processed):
    rows = []
    for build, gaps in (("B1", [30, 30, 30]), ("B2", [48, 50, 52]), ("B3", [20, 22, 24])):
        times = stamps(0, gaps)
        rows += [row(build, n, t) for n, t in zip([1, 2, 3, 4], times, strict=True)]
    rows.append(row("B1", 1, "20240318T130000.000000", modality="OT"))  # OT layers are ignored
    result = recoat.recoat_cycle(processed(rows), budget_fraction=0.5)
    assert {b: e["median_gap_s"] for b, e in result["per_build"].items()} == pytest.approx(
        {"B1": 30.0, "B2": 50.0, "B3": 22.0}
    )
    assert result["shortest_median_s"] == pytest.approx(22.0)
    assert result["budget_fraction"] == 0.5 and result["budget_s"] == pytest.approx(11.0)
    assert "layer index difference 1" in result["method"]

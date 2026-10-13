import json

import pytest
from tests.conftest import DETECT_CONFIG, detect_config_with
from typer.testing import CliRunner

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

from layerwise.cli import app  # noqa: E402
from layerwise.ingest.pipeline import run_ingest  # noqa: E402

pytestmark = pytest.mark.slow
runner = CliRunner()


def invoke(*args, config=DETECT_CONFIG):
    return runner.invoke(app, [*args, "--config", str(config), "--profile", "fixture"])


@pytest.fixture
def baseline_fold(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    assert invoke("baseline", "--fold", "0").exit_code == 0
    return fixture_data_root


def test_benchmark_writes_latency_and_recoat_cycle(baseline_fold):
    result = invoke("benchmark", "--run", "contrast/outer_0", "--runs", "3")
    assert result.exit_code == 0, result.output
    reports = baseline_fold / "reports/detect"
    records = json.loads((reports / "latency.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    rec = records[0]
    assert (rec["model"], rec["hardware"], rec["runtime"]) == ("contrast", "cpu", "opencv")
    assert rec["runs"] == 3 and rec["batch_size"] == 1 and rec["excludes"] == ["jpeg_decode"]
    assert rec["input_resolution"] == [320, 256] and rec["tile_size"] == 256
    assert rec["within_budget"] is True and rec["eligible_default"] is True
    cycle = json.loads((reports / "recoat_cycle.json").read_text(encoding="utf-8"))
    assert len(cycle["per_build"]) == 3
    assert cycle["budget_s"] == pytest.approx(cycle["shortest_median_s"] * 0.5)
    assert rec["budget_ms"] == pytest.approx(cycle["budget_s"] * 1000)


def test_budget_of_zero_still_writes_the_report_and_exits_four(baseline_fold, tmp_path):
    config = detect_config_with(tmp_path, "fixture", {"latency": {"budget_fraction": 0}})
    result = invoke("benchmark", "--run", "contrast/outer_0", "--runs", "2", config=config)
    assert result.exit_code == 4, result.output
    records = json.loads((baseline_fold / "reports/detect/latency.json").read_text("utf-8"))
    assert records[0]["within_budget"] is False and records[0]["eligible_default"] is False
    assert (baseline_fold / "reports/detect/recoat_cycle.json").exists()


def test_unknown_run_exits_two(baseline_fold):
    assert invoke("benchmark", "--run", "yolo/outer_0").exit_code == 2

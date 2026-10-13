import json

import pyarrow.parquet as pq
import pytest
from tests.conftest import DETECT_CONFIG
from typer.testing import CliRunner

pytest.importorskip("ultralytics")
pytest.importorskip("cv2")

from layerwise.cli import app  # noqa: E402
from layerwise.detect import data, metrics  # noqa: E402
from layerwise.detect.config import load_detect_config  # noqa: E402
from layerwise.detect.protocol import load_detection_evals  # noqa: E402
from layerwise.ingest.pipeline import run_ingest  # noqa: E402

pytestmark = pytest.mark.slow
runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, [*args, "--config", str(DETECT_CONFIG), "--profile", "fixture"])


@pytest.fixture
def both(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    assert invoke("baseline", "--fold", "0").exit_code == 0
    assert invoke("train", "--fold", "0").exit_code == 0
    return fixture_data_root


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_baseline_trains_nothing_and_matches_the_yolo_evaluation_setup(both):
    base = both / "runs/detect/contrast/outer_0"
    yolo = both / "runs/detect/yolo/outer_0"
    assert not list(base.rglob("*.pt"))
    assert read(base / "final/result.json")["trained"] is False
    a, b = read(base / "heldout_result.json"), read(yolo / "heldout_result.json")
    assert a["counts"]["images"] == b["counts"]["images"]
    assert a["counts"]["boxes"] == b["counts"]["boxes"]
    assert a["matching"] == b["matching"]
    assert a["label_source"] == b["label_source"]
    assert a["held_out_build"] == b["held_out_build"]


def test_baseline_threshold_comes_from_the_inner_validation_builds_only(both):
    cfg = load_detect_config(DETECT_CONFIG, "fixture")
    base = both / "runs/detect/contrast/outer_0"
    selection = read(base / "selection.json")
    plan = data.load_fold_plan(cfg)
    matches = []
    for inner in plan.outer_folds[0].inner:
        images = data.labeled_images(cfg, inner.validation_builds)
        path = base / f"inner_{inner.inner_fold}" / "contrast" / "val_detections.parquet"
        assert {r["image_id"] for r in pq.read_table(path).to_pylist()} <= {
            i.image_id for i in images
        }
        matches.append(metrics.match_images(load_detection_evals(path, images), 0.5))
    expected = metrics.best_f1_threshold(metrics.concat_matches(matches))
    held_out = plan.outer_folds[0].test_build
    assert held_out not in {b for i in plan.outer_folds[0].inner for b in i.train_builds}
    assert selection["threshold"] == pytest.approx(expected if expected is not None else 0.5)
    assert selection["epochs"] == 0


def test_report_lists_baseline_and_detector_on_the_same_images(both):
    result = invoke("report")
    assert result.exit_code == 0, result.output
    rows = pq.read_table(both / "reports/detect/comparison.parquet").to_pylist()
    pooled = {(r["model"], r["metric"]): r for r in rows if r["scope"] == "pooled"}
    assert pooled[("yolo", "recall")]["labeled_images"] == pooled[("contrast", "recall")][
        "labeled_images"
    ]
    assert all(r["partial"] for r in pooled.values())
    assert "missing folds: 1, 2" in (both / "reports/detect/comparison.md").read_text("utf-8")

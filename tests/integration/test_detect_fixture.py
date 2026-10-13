import json

import pyarrow.parquet as pq
import pytest
from tests.conftest import DETECT_CONFIG
from typer.testing import CliRunner

pytest.importorskip("ultralytics")

from layerwise.cli import app  # noqa: E402
from layerwise.ingest.pipeline import run_ingest  # noqa: E402

pytestmark = pytest.mark.slow
runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, [*args, "--config", str(DETECT_CONFIG), "--profile", "fixture"])


@pytest.fixture
def ingested(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    return fixture_data_root


def test_train_fold_zero_on_the_fixture(ingested):
    result = invoke("train", "--fold", "0")
    assert result.exit_code == 0, result.output
    run = ingested / "runs/detect/yolo/outer_0"
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    heldout = json.loads((run / "heldout_result.json").read_text(encoding="utf-8"))
    assert selection["winner"] == "yolov8n-scratch-t256"
    assert selection["epochs"] == 1
    assert set(selection["inner_validation_flags"].values()) == {"unreliable"}  # 20 images/build
    assert heldout["label_source"] == "aalto_manual_pb_boxes"
    assert heldout["matching"] == {"iou": 0.5, "one_to_one": True}
    assert heldout["counts"]["images"] == 20
    detections = pq.read_table(run / "heldout_detections.parquet")
    assert detections.column_names == [
        "detection_id", "image_id", "xmin", "ymin", "xmax", "ymax", "confidence", "outer_fold",
    ]  # fmt: skip
    for job in ("inner_0/yolov8n-scratch-t256", "inner_1/yolov8n-scratch-t256", "final"):
        for name in ("result.json", "manifest.json", "timing.json"):
            assert (run / job / name).exists(), f"{job}/{name}"
    assert (run / "heldout_manifest.json").exists() and (run / "heldout_timing.json").exists()
    for manifest in run.rglob("manifest.json"):
        assert str(ingested) not in manifest.read_text(encoding="utf-8")
    assert "unreliable" in result.output


def test_dry_run_writes_nothing(ingested):
    result = invoke("train", "--fold", "0", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "planned" in result.output
    assert not (ingested / "runs").exists()

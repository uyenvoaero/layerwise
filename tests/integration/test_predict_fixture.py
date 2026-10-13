import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image
from tests.conftest import DETECT_CONFIG
from typer.testing import CliRunner

pytest.importorskip("cv2")
pytest.importorskip("ultralytics")

from layerwise.cli import app  # noqa: E402
from layerwise.detect import representation as rep  # noqa: E402
from layerwise.ingest.pipeline import run_ingest  # noqa: E402

pytestmark = pytest.mark.slow
runner = CliRunner()


def invoke(*args):
    return runner.invoke(app, [*args, "--config", str(DETECT_CONFIG), "--profile", "fixture"])


@pytest.fixture
def baseline_fold(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    assert invoke("baseline", "--fold", "0").exit_code == 0
    return fixture_data_root


def test_predict_limit_writes_layers_detections_and_maps(baseline_fold):
    result = invoke("predict", "--run", "contrast/outer_0", "--limit", "3")
    assert result.exit_code == 0, result.output
    out = baseline_fold / "predictions/contrast/outer_0"
    layers = pq.read_table(out / "layers.parquet").to_pylist()
    detections = pq.read_table(out / "detections.parquet").to_pylist()
    assert len(layers) == 3
    assert [r["layer_index"] for r in layers] == sorted(r["layer_index"] for r in layers)
    for row in layers:
        assert (baseline_fold / row["map_path"]).exists()
        assert (baseline_fold / row["manifest_path"]).exists()
        assert row["status"] == "processed" and row["format_version"] == "1"
        assert row["model_id"] == "contrast/contrast"
    assert sum(r["n_detections"] for r in layers) == len(detections)


def test_maps_have_the_image_size_and_reproduce_the_detection_boxes(baseline_fold):
    result = invoke("predict", "--run", "contrast/outer_0", "--limit", "3", "--threshold", "0.3")
    assert result.exit_code == 0, result.output
    out = baseline_fold / "predictions/contrast/outer_0"
    layers = pq.read_table(out / "layers.parquet").to_pylist()
    detections = pq.read_table(out / "detections.parquet").to_pylist()
    images = {
        r["image_id"]: r["path"]
        for r in pq.read_table(
            baseline_fold / "processed/aalto/images.parquet", columns=["image_id", "path"]
        ).to_pylist()
    }
    for row in layers:
        with Image.open(baseline_fold / images[row["image_id"]]) as image:
            width, height = image.size
        defect_map, categories, version = rep.read_map(baseline_fold / row["map_path"])
        assert defect_map.shape == (1, height, width) and version == "1"
        mine = [d for d in detections if d["image_id"] == row["image_id"]]
        union = np.zeros((height, width), bool)
        for d in mine:
            assert d["confidence"] >= 0.3
            union[d["ymin"] : d["ymax"], d["xmin"] : d["xmax"]] = True
        assert np.array_equal(defect_map[0] >= np.float16(row["threshold"]), union)
        assert row["threshold"] == pytest.approx(0.3)


def test_unknown_run_exits_two(baseline_fold):
    result = invoke("predict", "--run", "yolo/outer_0")
    assert result.exit_code == 2

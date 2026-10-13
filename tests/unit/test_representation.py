import numpy as np
import pyarrow.parquet as pq
import pytest

from layerwise.detect import representation as rep
from layerwise.detect.models.base import Detection


def det(box, conf):
    return Detection(box=tuple(float(v) for v in box), confidence=conf)


def layer(image_id="IMG_1", detections=(), width=64, height=48, size_flag=False):
    return rep.LayerOutput(image_id, "B1", 1, size_flag, width, height, list(detections))


def test_map_has_one_channel_per_category_float16_in_unit_range():
    rows = rep.detection_rows(layer(detections=[det([4, 5, 20, 15], 0.8)]), threshold=0.0)
    defect_map = rep.rasterize(rows, 64, 48)
    assert defect_map.shape == (1, 48, 64) and defect_map.dtype == np.float16
    assert defect_map.min() >= 0 and defect_map.max() <= 1
    assert rep.CATEGORIES == ["recoating_phase_anomaly"]


def test_pixel_holds_the_highest_confidence_of_the_covering_boxes():
    rows = rep.detection_rows(
        layer(detections=[det([0, 0, 20, 20], 0.5), det([10, 10, 30, 30], 0.9)]), threshold=0.0
    )
    channel = rep.rasterize(rows, 64, 48)[0]
    assert channel[5, 5] == pytest.approx(0.5, abs=1e-3)  # only the first box
    assert channel[15, 15] == pytest.approx(0.9, abs=1e-3)  # both boxes: the higher one
    assert channel[25, 25] == pytest.approx(0.9, abs=1e-3)
    assert channel[40, 50] == 0  # elsewhere
    assert channel[19, 19] == pytest.approx(0.9, abs=1e-3) and channel[20, 5] == 0


def test_a_category_without_detections_is_an_all_zero_channel():
    rows = rep.detection_rows(layer(detections=[det([0, 0, 10, 10], 0.9)]), threshold=0.0)
    defect_map = rep.rasterize(rows, 64, 48, ["recoating_phase_anomaly", "melting_phase_anomaly"])
    assert defect_map.shape[0] == 2
    assert defect_map[0].max() > 0 and not defect_map[1].any()


def test_only_detections_at_or_above_the_threshold_are_stored_in_descending_order():
    lay = layer(
        detections=[det([0, 0, 5, 5], 0.2), det([6, 6, 12, 12], 0.9), det([20, 20, 30, 30], 0.5)]
    )
    rows = rep.detection_rows(lay, threshold=0.5)
    assert [r["detection_id"] for r in rows] == ["IMG_1#0", "IMG_1#1"]
    assert [r["confidence"] for r in rows] == pytest.approx([0.9, 0.5])
    assert all(r["source_class"] == "defects" for r in rows)


def test_boxes_are_integer_clipped_and_non_degenerate():
    lay = layer(detections=[det([-3.5, 2.2, 70.1, 60.0], 0.9), det([10, 10, 10.4, 30], 0.8)])
    rows = rep.detection_rows(lay, threshold=0.0)
    assert len(rows) == 2  # [10, 10.4] widens to a whole pixel by floor/ceil
    big = rows[0]
    assert (big["xmin"], big["ymin"], big["xmax"], big["ymax"]) == (0, 2, 64, 48)
    for r in rows:
        assert 0 <= r["xmin"] < r["xmax"] <= 64 and 0 <= r["ymin"] < r["ymax"] <= 48


def test_a_box_outside_the_image_is_dropped():
    rows = rep.detection_rows(layer(detections=[det([70, 5, 90, 20], 0.9)]), threshold=0.0)
    assert rows == []


def test_rebuilding_the_map_from_detections_parquet_equals_the_stored_map(tmp_path):
    layers = [
        layer("IMG_1", [det([4, 5, 20, 15], 0.8), det([10, 8, 40, 30], 0.6)]),
        layer("IMG_2", []),
    ]
    layer_rows, _ = rep.write_outputs(
        tmp_path / "pred",
        tmp_path,
        layers,
        model_id="m",
        manifest_path="runs/m.json",
        threshold=0.3,
    )
    table = pq.read_table(tmp_path / "pred" / "detections.parquet").to_pylist()
    for row in layer_rows:
        stored, categories, version = rep.read_map(tmp_path / row["map_path"])
        rebuilt = rep.rebuild_map(table, row["image_id"], 64, 48)
        assert np.array_equal(stored, rebuilt)
        assert categories == rep.CATEGORIES and version == "1"


def test_processed_layer_without_detections_has_a_row_and_unprocessed_layers_have_none(tmp_path):
    layer_rows, detections = rep.write_outputs(
        tmp_path / "pred",
        tmp_path,
        [layer("IMG_1", []), layer("IMG_3", [det([0, 0, 9, 9], 0.9)])],
        model_id="m",
        manifest_path="x",
        threshold=0.5,
    )
    by_id = {r["image_id"]: r for r in layer_rows}
    assert set(by_id) == {"IMG_1", "IMG_3"}  # IMG_2 was not processed: no row at all
    assert by_id["IMG_1"]["status"] == "processed" and by_id["IMG_1"]["n_detections"] == 0
    assert by_id["IMG_3"]["n_detections"] == 1 and len(detections) == 1


def test_files_are_identical_for_identical_inputs(tmp_path):
    lay = [layer("IMG_1", [det([4, 5, 20, 15], 0.8)])]
    for name in ("a", "b"):
        rep.write_outputs(
            tmp_path / name, tmp_path, lay, model_id="m", manifest_path="x", threshold=0.1
        )
    for relative in ("maps/IMG_1.npz", "detections.parquet"):  # layers hold the path
        assert (tmp_path / "a" / relative).read_bytes() == (tmp_path / "b" / relative).read_bytes()

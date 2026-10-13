import numpy as np
import pyarrow.parquet as pq

from layerwise.detect import representation as rep
from layerwise.detect.models.base import Detection

LAYERS = [
    ("image_id", "string", False),
    ("build_id", "string", False),
    ("layer_index", "int32", False),
    ("size_flag", "bool", False),
    ("status", "string", False),
    ("n_detections", "int32", False),
    ("map_path", "string", False),
    ("model_id", "string", False),
    ("manifest_path", "string", False),
    ("threshold", "double", False),
    ("format_version", "string", False),
]
DETECTIONS = [
    ("detection_id", "string", False),
    ("image_id", "string", False),
    ("xmin", "int32", False),
    ("ymin", "int32", False),
    ("xmax", "int32", False),
    ("ymax", "int32", False),
    ("source_class", "string", False),
    ("phase_category", "string", False),
    ("confidence", "float", False),
]


def spec(table):
    return [(f.name, str(f.type), f.nullable) for f in table.schema]


def write(tmp_path):
    layers = [
        rep.LayerOutput("IMG_B", "B1", 2, True, 64, 48, [Detection((1, 1, 9, 9), 0.7)]),
        rep.LayerOutput(
            "IMG_A",
            "B1",
            1,
            False,
            64,
            48,
            [Detection((0, 0, 5, 5), 0.4), Detection((9, 9, 30, 20), 0.9)],
        ),
    ]
    rep.write_outputs(
        tmp_path / "pred",
        tmp_path,
        layers,
        model_id="m/c",
        manifest_path="runs/x.json",
        threshold=0.3,
    )
    return tmp_path / "pred"


def test_layers_and_detections_columns_types_nullability_and_sort_order(tmp_path):
    out = write(tmp_path)
    layers = pq.read_table(out / "layers.parquet")
    detections = pq.read_table(out / "detections.parquet")
    assert spec(layers) == LAYERS
    assert spec(detections) == DETECTIONS
    assert layers.column("image_id").to_pylist() == ["IMG_A", "IMG_B"]
    assert detections.column("detection_id").to_pylist() == ["IMG_A#0", "IMG_A#1", "IMG_B#0"]
    assert layers.column("size_flag").to_pylist() == [False, True]
    assert set(layers.column("format_version").to_pylist()) == {"1"}
    assert all(0 <= c <= 1 for c in detections.column("confidence").to_pylist())


def test_defect_map_archive_has_the_contract_keys(tmp_path):
    out = write(tmp_path)
    with np.load(out / "maps" / "IMG_A.npz", allow_pickle=False) as archive:
        assert sorted(archive.files) == ["categories", "defect_map", "format_version"]
        assert archive["defect_map"].dtype == np.float16
        assert archive["defect_map"].shape == (1, 48, 64)
        assert str(archive["format_version"]) == "1"


def test_phase_category_vocabulary_contains_no_porosity(tmp_path):
    assert rep.CATEGORIES == ["recoating_phase_anomaly"]
    assert not any("porosity" in c for c in rep.CATEGORIES)
    detections = pq.read_table(write(tmp_path) / "detections.parquet")
    assert set(detections.column("phase_category").to_pylist()) == {"recoating_phase_anomaly"}
    assert set(detections.column("source_class").to_pylist()) == {"defects"}

import json
import re

import pyarrow.parquet as pq

from layerwise.ingest.pipeline import run_ingest

IMAGES = [
    ("image_id", "string", False),
    ("dataset", "string", False),
    ("build_id", "string", False),
    ("modality", "string", False),
    ("layer_index", "int32", False),
    ("timestamp", "string", True),
    ("z_mm", "double", True),
    ("path", "string", False),
    ("width", "int32", False),
    ("height", "int32", False),
    ("mode", "string", False),
    ("size_flag", "bool", False),
    ("label_status", "string", False),
    ("use_for_detection", "bool", False),
]
BOXES = [
    ("box_id", "string", False),
    ("image_id", "string", False),
    ("xmin", "int32", False),
    ("ymin", "int32", False),
    ("xmax", "int32", False),
    ("ymax", "int32", False),
    ("source_class", "string", False),
    ("phase_category", "string", False),
    ("fine_class", "string", True),
    ("clipped", "bool", False),
]
CLASS_MAPPING = [
    ("dataset", "string", False),
    ("source_class", "string", False),
    ("phase_category", "string", False),
    ("fine_class", "string", True),
    ("mapping_version", "string", False),
]
LAYER_PAIRS = [
    ("build_id", "string", False),
    ("layer_index", "int32", False),
    ("pb_image_id", "string", True),
    ("ot_image_id", "string", True),
    ("pair_status", "string", False),
]
FOLDS = [
    ("level", "string", False),
    ("outer_fold", "int32", False),
    ("inner_fold", "int32", True),
    ("build_id", "string", False),
    ("role", "string", False),
]


def columns(path):
    schema = pq.read_schema(path)
    return [(f.name, str(f.type), f.nullable) for f in schema]


def build(synthetic):
    synthetic.add_pb(2, boxes=[("defects", 1, 1, 9, 9)])
    synthetic.add_pb(1, boxes=[("defects", 20, 20, 30, 30), ("defects", 2, 2, 8, 8)])
    synthetic.add_pb(11, boxes=[("defects", 20, 20, 30, 30)])
    synthetic.add_pb(10)
    synthetic.add_ot(1, boxes=[("overheated", 3, 3, 40, 40)])


def test_images_and_boxes_schemas(ingest_cfg, synthetic):
    build(synthetic)
    run_ingest(ingest_cfg)
    assert columns(ingest_cfg.processed_dir / "images.parquet") == IMAGES
    assert columns(ingest_cfg.processed_dir / "boxes.parquet") == BOXES


def test_rows_sorted_by_key(ingest_cfg, synthetic):
    build(synthetic)
    run_ingest(ingest_cfg)
    ids = pq.read_table(ingest_cfg.processed_dir / "images.parquet")["image_id"].to_pylist()
    assert ids == sorted(ids)
    box_ids = pq.read_table(ingest_cfg.processed_dir / "boxes.parquet")["box_id"].to_pylist()
    keys = [(b.rsplit("#", 1)[0], int(b.rsplit("#", 1)[1])) for b in box_ids]
    assert keys == sorted(keys)
    assert len(set(box_ids)) == len(box_ids)


def test_box_invariants_and_mapping_columns(ingest_cfg, synthetic):
    build(synthetic)
    run_ingest(ingest_cfg)
    boxes = pq.read_table(ingest_cfg.processed_dir / "boxes.parquet").to_pylist()
    images = {
        r["image_id"]: r
        for r in pq.read_table(ingest_cfg.processed_dir / "images.parquet").to_pylist()
    }
    for box in boxes:
        image = images[box["image_id"]]
        assert 0 <= box["xmin"] < box["xmax"] <= image["width"]
        assert 0 <= box["ymin"] < box["ymax"] <= image["height"]
        assert box["fine_class"] is None


def test_other_table_schemas(ingest_cfg, synthetic):
    build(synthetic)
    run_ingest(ingest_cfg)
    assert columns(ingest_cfg.processed_dir / "class_mapping.parquet") == CLASS_MAPPING
    assert columns(ingest_cfg.processed_dir / "layer_pairs.parquet") == LAYER_PAIRS
    assert columns(ingest_cfg.processed_dir / "folds.parquet") == FOLDS


def test_json_outputs_have_no_timestamps(ingest_cfg, synthetic):
    build(synthetic)
    run_ingest(ingest_cfg)
    for name in ("manifest.json", "ingest_report.json"):
        text = (ingest_cfg.processed_dir / name).read_text(encoding="utf-8")
        data = json.loads(text)
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
        assert not re.search(r"(?i)timestamp|created|updated|generated_at", " ".join(data))


def test_manifest_and_report_keys(ingest_cfg, synthetic):
    build(synthetic)
    run_ingest(ingest_cfg)
    manifest = json.loads((ingest_cfg.processed_dir / "manifest.json").read_text("utf-8"))
    assert set(manifest) == {
        "dataset", "source", "code_version", "adapter_version", "input_hash", "config_hash",
        "config", "mapping_version", "seed", "outputs", "metrics",
    }  # fmt: skip
    assert set(manifest["metrics"]) == {
        "images", "labeled", "labeled_no_defects", "unlabeled", "boxes", "paired_layers", "folds",
    }  # fmt: skip
    report = json.loads((ingest_cfg.processed_dir / "ingest_report.json").read_text("utf-8"))
    for key in (
        "per_build", "boxes", "unparsed_files", "unknown_build_files", "orphan_xml",
        "duplicates", "failed_files",
    ):  # fmt: skip
        assert key in report

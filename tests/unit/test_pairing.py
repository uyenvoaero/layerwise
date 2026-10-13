import pytest

from layerwise.ingest import tables
from layerwise.ingest.pairing import pair_layers
from layerwise.ingest.sqlfiles import open_tables

BUILD = "SI383820211201123521"


def image(image_id, modality, layer, timestamp=None):
    return {
        "image_id": image_id,
        "dataset": "aalto",
        "build_id": BUILD,
        "modality": modality,
        "layer_index": layer,
        "timestamp": timestamp,
        "z_mm": None if modality == "PB" else 1.0,
        "path": f"aalto/{modality}/{image_id}.jpg",
        "width": 10,
        "height": 10,
        "mode": "L",
        "size_flag": False,
        "label_status": "unlabeled",
        "use_for_detection": False,
    }


def run(tmp_path, rows):
    tables.write_table(tmp_path / "images.parquet", rows, tables.IMAGES)
    con = open_tables(tmp_path, ["images"])
    try:
        return pair_layers(con)
    finally:
        con.close()


def test_paired_pb_only_and_ot_only(tmp_path):
    rows = [
        image("pb1", "PB", 1, "20240101T000001"),
        image("ot1", "OT", 1),
        image("pb2", "PB", 2, "20240101T000002"),
        image("ot3", "OT", 3),
    ]
    result = run(tmp_path, rows)
    pairs = {r["layer_index"]: r for r in result.table.to_pylist()}
    assert pairs[1]["pair_status"] == "paired"
    assert (pairs[1]["pb_image_id"], pairs[1]["ot_image_id"]) == ("pb1", "ot1")
    assert pairs[2]["pair_status"] == "pb_only" and pairs[2]["ot_image_id"] is None
    assert pairs[3]["pair_status"] == "ot_only" and pairs[3]["pb_image_id"] is None
    assert result.duplicates == []
    assert [r["layer_index"] for r in result.table.to_pylist()] == [1, 2, 3]


def test_duplicates_use_earliest_timestamp_then_filename(tmp_path):
    rows = [
        image("pbB", "PB", 5, "20240101T000009"),
        image("pbA", "PB", 5, "20240101T000001"),
        image("pbC", "PB", 5, "20240101T000009"),
        image("ot7", "OT", 5),
        image("ot8", "OT", 5),
    ]
    result = run(tmp_path, rows)
    (pair,) = result.table.to_pylist()
    assert (pair["pb_image_id"], pair["ot_image_id"]) == ("pbA", "ot7")
    dups = {(d["modality"], d["image_id"]): d["kept_image_id"] for d in result.duplicates}
    assert dups == {("PB", "pbB"): "pbA", ("PB", "pbC"): "pbA", ("OT", "ot8"): "ot7"}
    assert all(d["build_id"] == BUILD and d["layer_index"] == 5 for d in result.duplicates)


def test_pairing_is_by_layer_only_no_alignment_columns(tmp_path):
    result = run(tmp_path, [image("pb1", "PB", 1, "20240101T000001"), image("ot1", "OT", 1)])
    assert result.table.column_names == [
        "build_id", "layer_index", "pb_image_id", "ot_image_id", "pair_status",
    ]  # fmt: skip


def test_empty_images(tmp_path):
    result = run(tmp_path, [])
    assert result.table.num_rows == 0 and result.duplicates == []


@pytest.mark.parametrize("modality", ["PB", "OT"])
def test_single_modality_is_unpaired(tmp_path, modality):
    result = run(
        tmp_path, [image("x", modality, 1, "20240101T000001" if modality == "PB" else None)]
    )
    (pair,) = result.table.to_pylist()
    assert pair["pair_status"] == f"{modality.lower()}_only"

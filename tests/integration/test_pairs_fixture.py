import pyarrow.parquet as pq
from tests.conftest import BUILD_A

from layerwise.ingest.pipeline import run_ingest

PAIR_COLUMNS = ["build_id", "layer_index", "pb_image_id", "ot_image_id", "pair_status"]


def test_fixture_layer_pairs_columns_and_consistency(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    table = pq.read_table(ingest_cfg.processed_dir / "layer_pairs.parquet")
    assert table.column_names == PAIR_COLUMNS
    images = {
        r["image_id"]: r
        for r in pq.read_table(ingest_cfg.processed_dir / "images.parquet").to_pylist()
    }
    for row in table.to_pylist():
        if row["pair_status"] == "paired":
            pb, ot = images[row["pb_image_id"]], images[row["ot_image_id"]]
            assert pb["build_id"] == ot["build_id"] == row["build_id"]
            assert pb["layer_index"] == ot["layer_index"] == row["layer_index"]
            assert pb["modality"] == "PB" and ot["modality"] == "OT"
        elif row["pair_status"] == "pb_only":
            assert row["ot_image_id"] is None and row["pb_image_id"] in images
        else:
            assert row["pair_status"] == "ot_only" and row["pb_image_id"] is None


def test_unpaired_images_stay_in_images_and_are_counted(ingest_cfg, synthetic):
    synthetic.add_pb(1)
    synthetic.add_ot(1)
    synthetic.add_pb(2)
    synthetic.add_ot(9)
    outcome = run_ingest(ingest_cfg)
    pairs = pq.read_table(ingest_cfg.processed_dir / "layer_pairs.parquet").to_pylist()
    assert sorted(p["pair_status"] for p in pairs) == ["ot_only", "paired", "pb_only"]
    assert len(pq.read_table(ingest_cfg.processed_dir / "images.parquet")) == 4
    per_build = outcome.report["per_build"][BUILD_A]
    assert per_build["PB"]["unpaired"] == 1 and per_build["OT"]["unpaired"] == 1
    assert outcome.manifest["metrics"]["paired_layers"] == 1


def test_duplicate_layer_is_listed_in_report(ingest_cfg, synthetic):
    synthetic.add_pb(1, timestamp="20240318T140107.250")
    synthetic.add_pb(1, timestamp="20240318T140109.500")
    outcome = run_ingest(ingest_cfg)
    (dup,) = outcome.report["duplicates"]
    assert dup["modality"] == "PB" and dup["layer_index"] == 1
    assert outcome.report["per_build"][BUILD_A]["PB"]["duplicated"] == 1
    assert outcome.report["per_build"][BUILD_A]["PB"]["images"] == 2

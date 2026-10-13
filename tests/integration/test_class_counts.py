from collections import Counter

import pyarrow.parquet as pq

from layerwise.ingest.pipeline import run_ingest

MAPPING_COLUMNS = ["dataset", "source_class", "phase_category", "fine_class", "mapping_version"]


def test_class_mapping_table(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    table = pq.read_table(ingest_cfg.processed_dir / "class_mapping.parquet")
    assert table.column_names == MAPPING_COLUMNS
    rows = {r["source_class"]: r for r in table.to_pylist()}
    assert set(rows) == {"defects", "overheated"}
    assert all(r["mapping_version"] == "1" and r["dataset"] == "aalto" for r in rows.values())
    assert rows["defects"]["phase_category"] == "recoating_phase_anomaly"
    assert rows["overheated"]["phase_category"] == "melting_phase_anomaly"


def test_phase_counts_equal_mapped_source_class_counts(fixture_data_root, ingest_cfg):
    outcome = run_ingest(ingest_cfg)
    boxes = pq.read_table(ingest_cfg.processed_dir / "boxes.parquet").to_pylist()
    mapping = {
        r["source_class"]: r["phase_category"]
        for r in pq.read_table(ingest_cfg.processed_dir / "class_mapping.parquet").to_pylist()
    }
    by_class = Counter(b["source_class"] for b in boxes)
    by_phase = Counter(b["phase_category"] for b in boxes)
    for phase in set(mapping.values()):
        expected = sum(n for c, n in by_class.items() if mapping[c] == phase)
        assert by_phase[phase] == expected
    assert all(b["fine_class"] is None for b in boxes)
    # the report reconciles with the table
    report = outcome.report
    assert report["boxes_by_phase_category"] == dict(by_phase)
    assert {c: v["ingested"] for c, v in report["boxes"].items()} == dict(by_class)

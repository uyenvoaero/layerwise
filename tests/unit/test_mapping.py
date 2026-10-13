import pyarrow.parquet as pq
import pytest
from tests.conftest import BASE_CONFIG, REPO
from typer.testing import CliRunner

from layerwise.cli import app
from layerwise.errors import ConfigError, UnmappedClassError
from layerwise.ingest.mapping import load_mapping
from layerwise.ingest.pipeline import run_ingest

runner = CliRunner()


def test_aalto_mapping_values():
    mapping = load_mapping(REPO / "configs" / "class_mapping" / "aalto.yaml")
    assert mapping.version == "1" and mapping.dataset == "aalto"
    assert mapping.lookup("defects") == ("recoating_phase_anomaly", None)
    assert mapping.lookup("overheated") == ("melting_phase_anomaly", None)


def test_pb_and_ot_boxes_get_their_phase_category(ingest_cfg, synthetic):
    synthetic.add_pb(1, boxes=[("defects", 1, 1, 9, 9)])
    synthetic.add_ot(1, boxes=[("overheated", 2, 2, 20, 20)])
    run_ingest(ingest_cfg)
    boxes = pq.read_table(ingest_cfg.processed_dir / "boxes.parquet").to_pylist()
    assert {b["source_class"]: b["phase_category"] for b in boxes} == {
        "defects": "recoating_phase_anomaly",
        "overheated": "melting_phase_anomaly",
    }


def test_unmapped_class_exits_3_and_names_dataset_and_class(ingest_cfg, synthetic):
    synthetic.add_pb(1, boxes=[("scratch", 1, 1, 9, 9)])
    result = runner.invoke(app, ["--config", str(BASE_CONFIG), "--profile", "fixture", "ingest"])
    assert result.exit_code == 3
    assert "aalto" in result.output and "scratch" in result.output
    assert not ingest_cfg.processed_dir.exists()


def test_unmapped_class_raises_in_pipeline(ingest_cfg, synthetic):
    synthetic.add_ot(1, boxes=[("scratch", 1, 1, 9, 9)])
    with pytest.raises(UnmappedClassError, match="scratch"):
        run_ingest(ingest_cfg)


def test_porosity_entry_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        'version: "1"\ndataset: aalto\nclasses:\n'
        "  defects: {phase_category: recoating_phase_anomaly, fine_class: porosity}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="porosity"):
        load_mapping(bad)


def test_shipped_mapping_never_maps_to_porosity():
    mapping = load_mapping(REPO / "configs" / "class_mapping" / "aalto.yaml")
    assert all(fine != "porosity" for _, fine in mapping.classes.values())

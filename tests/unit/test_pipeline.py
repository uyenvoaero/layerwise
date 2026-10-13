import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tests.conftest import BASE_CONFIG, BUILD_A, BUILD_B
from typer.testing import CliRunner

from layerwise.cli import app
from layerwise.ingest import pipeline
from layerwise.ingest.pipeline import run_ingest

runner = CliRunner()


def build_small_dataset(syn):
    syn.add_pb(1, boxes=[("defects", 10, 10, 50, 50), ("defects", 60, 60, 90, 100)])
    syn.add_pb(2, boxes=[])
    syn.add_pb(3)
    syn.add_ot(1, boxes=[("overheated", 5, 5, 60, 60)])
    syn.add_ot(2)
    syn.add_pb(4, build=BUILD_B, boxes=[("defects", 1, 1, 9, 9)])


def table(cfg, name):
    return pq.read_table(cfg.processed_dir / f"{name}.parquet").to_pylist()


def snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


def invoke_ingest(*extra):
    return runner.invoke(
        app, ["--config", str(BASE_CONFIG), "--profile", "fixture", "ingest", *extra]
    )


def test_label_status_values_and_detection_flag(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    run_ingest(ingest_cfg)
    rows = {r["image_id"]: r for r in table(ingest_cfg, "images")}
    statuses = {r["label_status"] for r in rows.values()}
    assert statuses == {"labeled", "labeled_no_defects", "unlabeled"}
    for row in rows.values():
        assert row["use_for_detection"] == (row["label_status"] != "unlabeled")
    assert rows[synthetic.pb_stem(1)]["label_status"] == "labeled"
    assert rows[synthetic.pb_stem(2)]["label_status"] == "labeled_no_defects"
    assert rows[synthetic.pb_stem(3)]["label_status"] == "unlabeled"


def test_image_rows_carry_paths_sizes_and_modality_fields(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    run_ingest(ingest_cfg)
    rows = {r["image_id"]: r for r in table(ingest_cfg, "images")}
    pb = rows[synthetic.pb_stem(1)]
    assert pb["path"] == f"aalto/PB/{synthetic.pb_stem(1)}.jpg"
    assert (pb["width"], pb["height"], pb["mode"]) == (320, 256, "L")
    assert pb["timestamp"] == "20240318T140107.250" and pb["z_mm"] is None
    ot = rows[synthetic.ot_stem(1)]
    assert ot["layer_index"] == 1 and ot["timestamp"] is None and ot["z_mm"] == pytest.approx(39.4)
    assert pb["size_flag"] is False and ot["size_flag"] is False


def test_size_flag_when_size_differs_from_config(ingest_cfg, synthetic):
    synthetic.add_pb(1, size=(100, 80))
    synthetic.add_pb(2)
    run_ingest(ingest_cfg)
    rows = {r["image_id"]: r for r in table(ingest_cfg, "images")}
    assert rows[synthetic.pb_stem(1)]["size_flag"] is True
    assert rows[synthetic.pb_stem(2)]["size_flag"] is False


def test_sources_are_not_modified_or_copied(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in ingest_cfg.source_dir.rglob("*")
        if p.is_file()
    }
    run_ingest(ingest_cfg)
    after = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in ingest_cfg.source_dir.rglob("*")
        if p.is_file()
    }
    assert before == after
    assert not list(ingest_cfg.processed_dir.rglob("*.jpg"))
    assert not list(ingest_cfg.processed_dir.rglob("*.xml"))


def test_rerun_is_up_to_date_and_untouched(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    assert run_ingest(ingest_cfg).status == "written"
    first = snapshot(ingest_cfg.processed_dir)
    assert run_ingest(ingest_cfg).status == "up_to_date"
    assert snapshot(ingest_cfg.processed_dir) == first


def test_force_rewrites_byte_identical(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    run_ingest(ingest_cfg)
    first = {k: v[0] for k, v in snapshot(ingest_cfg.processed_dir).items()}
    assert run_ingest(ingest_cfg, force=True).status == "written"
    assert {k: v[0] for k, v in snapshot(ingest_cfg.processed_dir).items()} == first


def test_changed_input_triggers_rewrite(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    run_ingest(ingest_cfg)
    synthetic.add_pb(50, boxes=[("defects", 1, 1, 4, 4)])
    assert run_ingest(ingest_cfg).status == "written"
    assert len(table(ingest_cfg, "images")) == 7


def test_dry_run_writes_nothing(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    outcome = run_ingest(ingest_cfg, dry_run=True)
    assert outcome.status == "dry_run"
    assert outcome.report["per_build"]
    assert not ingest_cfg.processed_dir.exists()
    assert not list(ingest_cfg.processed_dir.parent.glob(".*"))


def test_interrupted_run_leaves_no_partial_output(ingest_cfg, synthetic, monkeypatch):
    build_small_dataset(synthetic)
    real = pipeline.write_table
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline, "write_table", flaky)
    with pytest.raises(RuntimeError):
        run_ingest(ingest_cfg)
    assert not ingest_cfg.processed_dir.exists()
    assert not list(ingest_cfg.processed_dir.parent.glob(".*"))


def test_interrupted_rerun_keeps_previous_output(ingest_cfg, synthetic, monkeypatch):
    build_small_dataset(synthetic)
    run_ingest(ingest_cfg)
    first = {k: v[0] for k, v in snapshot(ingest_cfg.processed_dir).items()}

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(pipeline, "write_table", crash)
    with pytest.raises(RuntimeError):
        run_ingest(ingest_cfg, force=True)
    assert {k: v[0] for k, v in snapshot(ingest_cfg.processed_dir).items()} == first


def test_corrupt_files_are_recorded_and_ingest_continues(ingest_cfg, synthetic):
    for layer in range(1, 41):
        synthetic.add_pb(layer, boxes=[("defects", 1, 1, 8, 8)])
    synthetic.add_corrupt_jpeg("PB", layer=900)
    synthetic.add_corrupt_xml(layer=901)
    outcome = run_ingest(ingest_cfg)
    failed = outcome.report["failed_files"]
    assert {(f["kind"], Path(f["path"]).name) for f in failed} == {
        ("image", f"{synthetic.pb_stem(900)}.jpg"),
        ("xml", f"{synthetic.pb_stem(901)}.xml"),
    }
    ids = {r["image_id"] for r in table(ingest_cfg, "images")}
    assert synthetic.pb_stem(900) not in ids  # unreadable image is not ingested
    assert len(ids) == 41  # 40 good + the image whose XML is malformed
    row = next(r for r in table(ingest_cfg, "images") if r["image_id"] == synthetic.pb_stem(901))
    assert row["label_status"] == "unlabeled"


def test_exit_4_above_failure_threshold(ingest_cfg, synthetic):
    for layer in range(1, 21):
        synthetic.add_pb(layer)
    synthetic.add_corrupt_jpeg("PB", layer=900)
    synthetic.add_corrupt_jpeg("PB", layer=901)  # 2 of 22 files fail: 9% > 5%
    result = invoke_ingest()
    assert result.exit_code == 4
    assert not ingest_cfg.processed_dir.exists()


def test_exit_0_below_failure_threshold(ingest_cfg, synthetic):
    for layer in range(1, 41):
        synthetic.add_pb(layer)
    synthetic.add_corrupt_jpeg("PB", layer=900)  # 1 of 41 files fails: 2.4% < 5%
    result = invoke_ingest()
    assert result.exit_code == 0, result.output
    assert (ingest_cfg.processed_dir / "ingest_report.json").exists()


def test_report_lists_unparsed_unknown_and_orphans(ingest_cfg, synthetic):
    synthetic.add_pb(1, boxes=[("defects", 1, 1, 8, 8)])
    synthetic.add_bad_name("PB", "garbage")
    synthetic.add_pb(2, build="SI999999999999999999")
    synthetic.add_orphan_xml("PB", synthetic.pb_stem(77))
    for layer in range(10, 60):
        synthetic.add_pb(layer)
    report = run_ingest(ingest_cfg).report
    assert [Path(p).name for p in report["unparsed_files"]] == ["garbage.jpg"]
    assert len(report["unknown_build_files"]) == 1
    assert [Path(p).name for p in report["orphan_xml"]] == [f"{synthetic.pb_stem(77)}.xml"]


def test_report_counts_boxes_and_per_build(ingest_cfg, synthetic):
    synthetic.add_pb(1, boxes=[("defects", -4, 10, 50, 50), ("defects", 5, 5, 5, 9)])
    synthetic.add_ot(1, boxes=[("overheated", 5, 5, 60, 60)])
    report = run_ingest(ingest_cfg).report
    assert report["boxes"]["defects"] == {"ingested": 1, "clipped": 1, "dropped": 1}
    assert report["boxes"]["overheated"] == {"ingested": 1, "clipped": 0, "dropped": 0}
    pb = report["per_build"][BUILD_A]["PB"]
    assert pb["images"] == 1 and pb["labeled"] == 1
    assert report["per_build"][BUILD_A]["OT"]["images"] == 1


def test_manifest_and_report_are_json_files_without_timestamps(ingest_cfg, synthetic):
    build_small_dataset(synthetic)
    run_ingest(ingest_cfg)
    for name in ("manifest.json", "ingest_report.json"):
        text = (ingest_cfg.processed_dir / name).read_text(encoding="utf-8")
        assert text.endswith("}\n")
        json.loads(text)
        assert str(ingest_cfg.data_root) not in text  # paths relative to DATA_ROOT only
    manifest = json.loads((ingest_cfg.processed_dir / "manifest.json").read_text("utf-8"))
    assert manifest["source"]["license"] == "CC-BY-4.0"
    assert manifest["metrics"]["images"] == 6
    assert manifest["metrics"]["boxes"] == 4
    assert manifest["seed"] is None

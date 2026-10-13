import json

import pyarrow.parquet as pq
from tests.conftest import FIXTURE_SOURCE

from layerwise.ingest.pipeline import run_ingest

MB = 1024 * 1024


def read(cfg, name):
    return pq.read_table(cfg.processed_dir / f"{name}.parquet").to_pylist()


def test_fixture_is_small_and_attributed():
    files = [p for p in FIXTURE_SOURCE.rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in files) < 5 * MB
    attribution = (FIXTURE_SOURCE / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "10.5281/zenodo.14996805" in attribution and "CC-BY-4.0" in attribution
    assert len(list((FIXTURE_SOURCE / "PB").glob("*.jpg"))) == 60
    assert len(list((FIXTURE_SOURCE / "OT").glob("*.jpg"))) == 60


def test_ingest_fixture_counts_and_sizes(fixture_data_root, ingest_cfg):
    outcome = run_ingest(ingest_cfg)
    images = read(ingest_cfg, "images")
    assert len(images) == 120
    assert len({r["build_id"] for r in images}) == 3
    assert not any(r["size_flag"] for r in images)
    assert all(r["label_status"] == "labeled" for r in images)
    by_id = {r["image_id"]: r for r in images}
    for box in read(ingest_cfg, "boxes"):
        image = by_id[box["image_id"]]
        assert 0 <= box["xmin"] < box["xmax"] <= image["width"]
        assert 0 <= box["ymin"] < box["ymax"] <= image["height"]
    assert outcome.manifest["source"]["license"] == "CC-BY-4.0"
    manifest = json.loads((ingest_cfg.processed_dir / "manifest.json").read_text("utf-8"))
    assert manifest["source"]["license"] == "CC-BY-4.0"


def test_ingest_fixture_rerun_is_byte_identical(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    first = {p.name: p.read_bytes() for p in ingest_cfg.processed_dir.iterdir()}
    run_ingest(ingest_cfg, force=True)
    assert {p.name: p.read_bytes() for p in ingest_cfg.processed_dir.iterdir()} == first


def test_outputs_stay_outside_the_fixture_directory(fixture_data_root, ingest_cfg):
    before = sorted(p.relative_to(FIXTURE_SOURCE) for p in FIXTURE_SOURCE.rglob("*"))
    run_ingest(ingest_cfg)
    assert sorted(p.relative_to(FIXTURE_SOURCE) for p in FIXTURE_SOURCE.rglob("*")) == before

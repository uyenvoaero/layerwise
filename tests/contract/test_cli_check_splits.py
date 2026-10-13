import pyarrow as pa
import pyarrow.parquet as pq
from tests.conftest import BASE_CONFIG
from typer.testing import CliRunner

from layerwise.cli import app
from layerwise.ingest.pipeline import run_ingest

runner = CliRunner()


def check_splits():
    return runner.invoke(
        app, ["--config", str(BASE_CONFIG), "--profile", "fixture", "check-splits"]
    )


def test_check_splits_ok_prints_per_role_counts(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    result = check_splits()
    assert result.exit_code == 0, result.output
    assert "3 outer folds" in result.output
    assert "2 inner folds" in result.output
    for word in ("test", "train", "validation", "images", "labeled", "boxes"):
        assert word in result.output


def test_check_splits_fails_on_tampered_folds(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    path = ingest_cfg.processed_dir / "folds.parquet"
    rows = pq.read_table(path).to_pylist()
    first_outer = next(r for r in rows if r["level"] == "outer" and r["role"] == "test")
    rows.append({**first_outer, "role": "train"})  # the same build now has two roles
    pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(path)), path)
    result = check_splits()
    assert result.exit_code == 1
    assert first_outer["build_id"] in result.output
    assert f"outer fold {first_outer['outer_fold']}" in result.output


def test_check_splits_without_ingest_output_is_an_error(fixture_data_root):
    result = check_splits()
    assert result.exit_code == 2

import importlib.util
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.conftest import DETECT_CONFIG, needs_vision
from typer.testing import CliRunner

from layerwise.cli import app
from layerwise.ingest.pipeline import run_ingest

runner = CliRunner()
GLOBAL = ["--config", str(DETECT_CONFIG), "--profile", "fixture"]


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def help_of(command: str) -> str:
    return plain(runner.invoke(app, [command, "--help"]).output)


@pytest.fixture
def ingested(fixture_data_root, ingest_cfg):
    run_ingest(ingest_cfg)
    return fixture_data_root


def test_command_names():
    out = plain(runner.invoke(app, ["--help"]).output)
    for command in ("train", "baseline", "predict", "benchmark", "report"):
        assert command in out


def test_option_names_per_command():
    for option in ("--model", "--fold", "--dry-run", "--repro", "--config", "--profile"):
        assert option in help_of("train")
    for option in ("--fold", "--dry-run", "--repro"):
        assert option in help_of("baseline")
    for option in ("--run", "--build", "--limit", "--threshold", "--config", "--profile"):
        assert option in help_of("predict")
    for option in ("--run", "--runs", "--config", "--profile"):
        assert option in help_of("benchmark")
    for option in ("--config", "--profile"):
        assert option in help_of("report")


@needs_vision
def test_global_options_work_before_or_after_the_command(ingested):
    after = runner.invoke(app, ["train", "--dry-run", *GLOBAL])
    before = runner.invoke(app, [*GLOBAL, "train", "--dry-run"])
    assert after.exit_code == 0 and before.exit_code == 0, (after.output, before.output)
    assert after.output == before.output and "planned" in after.output


@needs_vision
def test_exit_0_for_a_dry_run_and_nothing_is_written(ingested):
    result = runner.invoke(app, ["baseline", "--dry-run", *GLOBAL])
    assert result.exit_code == 0 and not (ingested / "runs").exists()


@needs_vision
def test_exit_1_when_a_build_has_two_roles_in_a_fold(ingested, ingest_cfg):
    path = ingest_cfg.processed_dir / "folds.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.append({**rows[0], "role": "test"})
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    result = runner.invoke(app, ["train", "--dry-run", *GLOBAL])
    assert result.exit_code == 1 and "roles" in result.output


@needs_vision
def test_exit_2_for_missing_data_root(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    result = runner.invoke(app, ["train", "--dry-run", *GLOBAL])
    assert result.exit_code == 2 and "DATA_ROOT" in result.output


@needs_vision
def test_exit_2_for_missing_ingest_output(data_root):
    result = runner.invoke(app, ["train", "--dry-run", *GLOBAL])
    assert result.exit_code == 2 and "ingest" in result.output


@needs_vision
def test_exit_2_for_an_unknown_model(ingested):
    result = runner.invoke(app, ["train", "--model", "resnet", *GLOBAL])
    assert result.exit_code == 2


def test_exit_2_with_install_hint_when_the_vision_extra_is_missing(ingested, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "ultralytics" else real(name)
    )
    result = runner.invoke(app, ["train", "--dry-run", *GLOBAL])
    assert result.exit_code == 2 and "uv sync --extra vision" in result.output


@needs_vision
def test_exit_2_for_report_without_results_and_for_unknown_run(ingested):
    report = runner.invoke(app, ["report", *GLOBAL])
    assert report.exit_code == 2 and "no finished" in report.output
    for command in ("predict", "benchmark"):
        result = runner.invoke(app, [command, "--run", "yolo/outer_0", *GLOBAL])
        assert result.exit_code == 2


@needs_vision
def test_exit_3_for_unreadable_labeled_images_above_the_failure_threshold(ingested):
    images = sorted((ingested / "aalto" / "PB").glob("*.jpg"))
    for path in images[:10]:  # of 60 labeled PB images: above the 5% threshold
        path.write_bytes(b"not a jpeg")
    result = runner.invoke(app, ["baseline", "--fold", "0", *GLOBAL])
    assert result.exit_code == 3 and "unreadable" in result.output
    assert not (ingested / "runs").exists()

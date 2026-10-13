import re

from tests.conftest import BASE_CONFIG
from typer.testing import CliRunner

from layerwise.cli import app

runner = CliRunner()
GLOBAL = ["--config", str(BASE_CONFIG), "--profile", "fixture"]


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_command_names():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "check-splits", "make-fixture"):
        assert command in plain(result.output)


def test_global_option_names_and_profile_values():
    out = plain(runner.invoke(app, ["--help"]).output)
    assert "--config" in out and "--profile" in out
    for profile in ("local", "aws", "fixture"):
        assert profile in out
    bad = runner.invoke(app, ["--profile", "nope", "ingest"])
    assert bad.exit_code != 0


def test_ingest_option_names():
    out = plain(runner.invoke(app, ["ingest", "--help"]).output)
    assert "--force" in out and "--dry-run" in out


def test_exit_0_and_up_to_date(fixture_data_root):
    first = runner.invoke(app, [*GLOBAL, "ingest"])
    assert first.exit_code == 0, first.output
    second = runner.invoke(app, [*GLOBAL, "ingest"])
    assert second.exit_code == 0 and "up to date" in second.output
    forced = runner.invoke(app, [*GLOBAL, "ingest", "--force"])
    assert forced.exit_code == 0 and "up to date" not in forced.output


def test_dry_run_exits_0_and_writes_nothing(fixture_data_root, ingest_cfg):
    result = runner.invoke(app, [*GLOBAL, "ingest", "--dry-run"])
    assert result.exit_code == 0 and "images: 120" in result.output
    assert not ingest_cfg.processed_dir.exists()


def test_exit_2_for_missing_data_root(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    result = runner.invoke(app, [*GLOBAL, "ingest"])
    assert result.exit_code == 2 and "DATA_ROOT" in result.output


def test_exit_2_for_missing_source_folder(data_root):
    result = runner.invoke(app, [*GLOBAL, "ingest"])
    assert result.exit_code == 2


def test_exit_3_for_unmapped_class(synthetic):
    synthetic.add_pb(1, boxes=[("scratch", 1, 1, 9, 9)])
    assert runner.invoke(app, [*GLOBAL, "ingest"]).exit_code == 3


def test_exit_4_for_input_errors_above_threshold(synthetic):
    synthetic.add_pb(1)
    synthetic.add_corrupt_jpeg("PB", layer=900)
    assert runner.invoke(app, [*GLOBAL, "ingest"]).exit_code == 4


def test_exit_1_for_make_fixture_with_too_few_labeled_images(synthetic):
    # Selection fails before anything is written, so tests/fixtures/ is left untouched.
    synthetic.add_pb(1, boxes=[("defects", 1, 1, 9, 9)])
    result = runner.invoke(app, [*GLOBAL, "make-fixture"])
    assert result.exit_code == 1
    assert "labeled" in result.output

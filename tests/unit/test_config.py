from pathlib import Path

import pytest

from layerwise.config import load_config
from layerwise.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "configs" / "ingest" / "base.yaml"


def test_profile_overrides_base(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    cfg = load_config(BASE, "fixture")
    assert cfg.raw["source_dir"] == "aalto"
    assert cfg.expected_size["PB"] == (320, 256)
    assert cfg.expected_size["OT"] == (320, 320)
    assert cfg.mlflow_enabled is False
    # keys not overridden come from base
    assert cfg.dataset == "aalto"
    assert len(cfg.known_builds) == 3


def test_local_profile_keeps_full_size(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    cfg = load_config(BASE, "local")
    assert cfg.expected_size["PB"] == (1280, 1024)
    assert cfg.expected_size["OT"] == (2000, 2000)
    assert cfg.source_dir == tmp_path / "aalto" / "Original Images and manual labels"


def test_data_root_read_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert load_config(BASE, "fixture").data_root == tmp_path


def test_missing_data_root_is_config_error(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(ConfigError, match="DATA_ROOT"):
        load_config(BASE, "fixture")


def test_data_root_not_a_directory_is_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "missing"))
    with pytest.raises(ConfigError, match="DATA_ROOT"):
        load_config(BASE, "fixture")


def test_bad_yaml_is_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    bad = tmp_path / "base.yaml"
    bad.write_text("dataset: [unclosed\n", encoding="utf-8")
    (tmp_path / "local.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad, "local")


def test_unknown_profile_is_config_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    with pytest.raises(ConfigError, match="not found"):
        load_config(BASE, "nope")

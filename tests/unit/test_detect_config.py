import importlib.util

import pytest
import yaml
from tests.conftest import DETECT_CONFIG

from layerwise.detect.config import check_vision_extra, load_detect_config
from layerwise.errors import ConfigError

BASE = DETECT_CONFIG
REPO = DETECT_CONFIG.parents[2]


def test_profile_merges_over_base(data_root):
    cfg = load_detect_config(BASE, "local")
    assert cfg.tiling.tile_size == 256  # local override
    assert cfg.tiling.min_box_visible == 0.6  # base value kept
    assert cfg.train["max_epochs"] == 1
    assert cfg.train["patience"] == 1
    assert cfg.seed == 0


def test_ingest_config_resolved_from_repo_root_with_named_profile(data_root):
    cfg = load_detect_config(BASE, "fixture")
    assert cfg.ingest.dataset == "aalto"
    assert cfg.ingest.expected_size["PB"] == (320, 256)  # ingest fixture profile
    assert cfg.ingest.repo_root == REPO
    assert cfg.ingest.data_root == data_root


def test_directories_are_relative_to_data_root(data_root):
    cfg = load_detect_config(BASE, "local")
    assert cfg.runs_dir == data_root / "runs/detect"
    assert cfg.staging_dir == data_root / "staging"
    assert cfg.predictions_dir == data_root / "predictions"
    assert cfg.reports_dir == data_root / "reports/detect"
    assert cfg.weights_dir == data_root / "weights"
    assert cfg.relative_to_root(cfg.runs_dir / "yolo") == "runs/detect/yolo"


def test_candidates_carry_tile_size_and_default_overlap(data_root):
    cfg = load_detect_config(BASE, "aws")
    candidates = cfg.candidates("yolo")
    assert [c.id for c in candidates] == [
        "yolov8n-t640",
        "yolov8n-t1024",
        "yolov8s-t640",
        "yolov8s-t1024",
    ]
    assert candidates[1].tile_size == 1024
    assert candidates[0].overlap == cfg.tiling.overlap
    assert cfg.tiling_for(candidates[1]).tile_size == 1024


def test_more_than_four_candidates_is_a_config_error(data_root, tmp_path):
    raw = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    raw["models"]["yolo"]["candidates"] = [{"id": f"c{i}", "tile_size": 640} for i in range(5)]
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(raw), encoding="utf-8")
    (tmp_path / "local.yaml").write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="at most 4"):
        load_detect_config(base, "local")


def test_unknown_model_is_a_config_error(data_root):
    cfg = load_detect_config(BASE, "local")
    with pytest.raises(ConfigError, match="unknown model"):
        cfg.candidates("resnet")


def test_missing_data_root_is_a_config_error(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(ConfigError, match="DATA_ROOT"):
        load_detect_config(BASE, "local")


def test_missing_vision_extra_gives_install_hint(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "ultralytics" else real(name)
    )
    with pytest.raises(ConfigError, match=r"uv sync --extra vision"):
        check_vision_extra()

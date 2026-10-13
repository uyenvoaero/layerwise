import importlib.util
import shutil
from pathlib import Path

import pytest
from PIL import Image

from layerwise.config import IngestConfig, load_config
from layerwise.detect.config import VISION_MODULES

REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs" / "ingest" / "base.yaml"
DETECT_CONFIG = REPO / "configs" / "detect" / "base.yaml"
FIXTURE_SOURCE = REPO / "tests" / "fixtures" / "aalto"

BUILD_A = "SI383820211201123521"
BUILD_B = "SI383820240226095904"
BUILD_C = "SI383820240318120348"
UNKNOWN_BUILD = "SI999999999999999999"

needs_vision = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in VISION_MODULES),
    reason="needs the optional vision extra (uv sync --extra vision)",
)

PB_SIZE = (320, 256)  # (width, height), matches configs/ingest/fixture.yaml
OT_SIZE = (320, 320)


def _xml(filename: str, size: tuple[int, int], boxes: list[tuple] | None) -> str:
    objects = "".join(
        f"<object><name>{name}</name><pose>Unspecified</pose><truncated>0</truncated>"
        f"<difficult>0</difficult><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
        for name, x0, y0, x1, y1 in (boxes or [])
    )
    return (
        "<annotation><folder>PB</folder>"
        f"<filename>{filename}</filename><path>C:\\Users\\author\\{filename}</path>"
        f"<source><database>Unknown</database></source><size><width>{size[0]}</width>"
        f"<height>{size[1]}</height><depth>1</depth></size><segmented>0</segmented>"
        f"{objects}</annotation>"
    )


class SyntheticDataset:
    """Builds a tiny Aalto-like folder tree (PB, PB_label, OT, OT_label) under a source dir."""

    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        for folder in ("PB", "PB_label", "OT", "OT_label"):
            (source_dir / folder).mkdir(parents=True, exist_ok=True)

    def _jpeg(self, folder: str, stem: str, size: tuple[int, int]) -> Path:
        path = self.source_dir / folder / f"{stem}.jpg"
        Image.new("L", size, 128).save(path, "JPEG")
        return path

    def _label(self, folder: str, stem: str, size, boxes) -> None:
        text = _xml(f"{stem}.jpg", size, boxes)
        (self.source_dir / f"{folder}_label" / f"{stem}.xml").write_text(text, encoding="utf-8")

    def pb_stem(self, layer: int, build: str = BUILD_A, timestamp: str = "20240318T140107.250"):
        return f"{build}_{layer:05d}_{timestamp}"

    def ot_stem(self, layer: int, build: str = BUILD_A, z: str = "039_400"):
        return f"{build}_{layer}_{z}_Int_32F"

    def add_pb(self, layer, boxes=None, build=BUILD_A, size=PB_SIZE, timestamp=None, xml_size=None):
        """boxes: None = no XML (unlabeled), [] = XML without boxes, else (name, x0, y0, x1, y1)."""
        stem = self.pb_stem(layer, build, timestamp or "20240318T140107.250")
        self._jpeg("PB", stem, size)
        if boxes is not None:
            self._label("PB", stem, xml_size or size, boxes)
        return stem

    def add_ot(self, layer, boxes=None, build=BUILD_A, size=OT_SIZE, z="039_400", xml_size=None):
        stem = self.ot_stem(layer, build, z)
        self._jpeg("OT", stem, size)
        if boxes is not None:
            self._label("OT", stem, xml_size or size, boxes)
        return stem

    def add_orphan_xml(self, modality: str, stem: str) -> None:
        self._label(modality, stem, PB_SIZE, [("defects", 1, 1, 5, 5)])

    def add_bad_name(self, modality: str = "PB", name: str = "not_a_valid_name") -> str:
        self._jpeg(modality, name, (16, 16))
        return f"{name}.jpg"

    def add_corrupt_jpeg(self, modality="PB", layer=900, build=BUILD_A) -> str:
        stem = self.pb_stem(layer, build) if modality == "PB" else self.ot_stem(layer, build)
        (self.source_dir / modality / f"{stem}.jpg").write_bytes(b"this is not a jpeg")
        return stem

    def add_corrupt_xml(self, layer=901, build=BUILD_A, size=PB_SIZE) -> str:
        stem = self.add_pb(layer, build=build, size=size)
        (self.source_dir / "PB_label" / f"{stem}.xml").write_text("<annotation><oops", "utf-8")
        return stem


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data_root"
    root.mkdir()
    monkeypatch.setenv("DATA_ROOT", str(root))
    return root


@pytest.fixture
def ingest_cfg(data_root) -> IngestConfig:
    """Fixture-profile config (scaled expected sizes, MLflow off) over an empty DATA_ROOT."""
    return load_config(BASE_CONFIG, "fixture")


@pytest.fixture
def synthetic(ingest_cfg) -> SyntheticDataset:
    return SyntheticDataset(ingest_cfg.source_dir)


@pytest.fixture
def fixture_data_root(data_root) -> Path:
    """DATA_ROOT holding a copy of tests/fixtures/aalto; outputs never go under tests/fixtures/."""
    shutil.copytree(FIXTURE_SOURCE, data_root / "aalto")
    return data_root


def detect_config_with(tmp_path: Path, profile: str, overrides: dict) -> Path:
    """Detect base config copied to `tmp_path`, its `profile` file merged with `overrides`."""
    import yaml

    from layerwise.config import _merge

    base = yaml.safe_load(DETECT_CONFIG.read_text(encoding="utf-8"))
    base["ingest_config"] = str(REPO / "configs" / "ingest" / "base.yaml")
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    original = yaml.safe_load((DETECT_CONFIG.parent / f"{profile}.yaml").read_text("utf-8"))
    (tmp_path / f"{profile}.yaml").write_text(
        yaml.safe_dump(_merge(original, overrides)), encoding="utf-8"
    )
    return tmp_path / "base.yaml"

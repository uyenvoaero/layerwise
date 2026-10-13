import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layerwise.config import IngestConfig, _find_repo_root, _merge, _read_yaml, load_config
from layerwise.errors import ConfigError

DEFAULT_DETECT_CONFIG = "configs/detect/base.yaml"
MAX_CANDIDATES = 4
MODELS = ("yolo", "frcnn", "contrast")
VISION_MODULES = ("torch", "torchvision", "ultralytics", "albumentations", "cv2")
REQUIRED_KEYS = (
    "ingest_config",
    "ingest_profile",
    "runs_dir",
    "staging_dir",
    "predictions_dir",
    "reports_dir",
    "weights_dir",
    "seed",
    "tiling",
    "matching",
    "size_buckets",
    "eval_min_confidence",
    "repro_tolerance",
    "min_validation_images",
    "models",
    "train",
    "latency",
    "tracking",
)


@dataclass(frozen=True)
class TilingParams:
    tile_size: int
    overlap: int
    min_box_visible: float
    negative_tile_ratio: float
    nms_iou: float


@dataclass(frozen=True)
class Candidate:
    """One search configuration of a model."""

    id: str
    tile_size: int
    overlap: int
    weights: str | None = None


@dataclass(frozen=True)
class DetectConfig:
    """Resolved detect configuration. `raw` keeps paths relative, as written in the YAML."""

    raw: dict[str, Any]
    ingest: IngestConfig
    profile: str

    @property
    def data_root(self) -> Path:
        return self.ingest.data_root

    @property
    def repo_root(self) -> Path:
        return self.ingest.repo_root

    @property
    def runs_dir(self) -> Path:
        return self.data_root / self.raw["runs_dir"]

    @property
    def staging_dir(self) -> Path:
        return self.data_root / self.raw["staging_dir"]

    @property
    def predictions_dir(self) -> Path:
        return self.data_root / self.raw["predictions_dir"]

    @property
    def reports_dir(self) -> Path:
        return self.data_root / self.raw["reports_dir"]

    @property
    def weights_dir(self) -> Path:
        return self.data_root / self.raw["weights_dir"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def tiling(self) -> TilingParams:
        t = self.raw["tiling"]
        return TilingParams(
            tile_size=int(t["tile_size"]),
            overlap=int(t["overlap"]),
            min_box_visible=float(t["min_box_visible"]),
            negative_tile_ratio=float(t["negative_tile_ratio"]),
            nms_iou=float(t["nms_iou"]),
        )

    @property
    def match_iou(self) -> float:
        return float(self.raw["matching"]["iou"])

    @property
    def size_buckets(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.raw["size_buckets"].items()}

    @property
    def eval_min_confidence(self) -> float:
        return float(self.raw["eval_min_confidence"])

    @property
    def repro_tolerance(self) -> float:
        return float(self.raw["repro_tolerance"])

    @property
    def min_validation_images(self) -> int:
        return int(self.raw["min_validation_images"])

    @property
    def train(self) -> dict[str, Any]:
        return dict(self.raw["train"])

    @property
    def latency(self) -> dict[str, Any]:
        return dict(self.raw["latency"])

    @property
    def mlflow_enabled(self) -> bool:
        return bool(self.raw["tracking"].get("mlflow", False))

    @property
    def mlflow_dir(self) -> Path:
        return self.data_root / self.raw["tracking"].get("dir", "mlruns")

    def model_options(self, model: str) -> dict[str, Any]:
        """Model settings other than the candidate list (learning rate, contrast parameters)."""
        self._check_model(model)
        return {k: v for k, v in self.raw["models"][model].items() if k != "candidates"}

    def candidates(self, model: str) -> list[Candidate]:
        self._check_model(model)
        out = []
        for entry in self.raw["models"][model]["candidates"]:
            out.append(
                Candidate(
                    id=str(entry["id"]),
                    tile_size=int(entry.get("tile_size", self.tiling.tile_size)),
                    overlap=int(entry.get("overlap", self.tiling.overlap)),
                    weights=entry.get("weights"),
                )
            )
        return out

    def tiling_for(self, candidate: Candidate) -> TilingParams:
        base = self.tiling
        return TilingParams(
            tile_size=candidate.tile_size,
            overlap=candidate.overlap,
            min_box_visible=base.min_box_visible,
            negative_tile_ratio=base.negative_tile_ratio,
            nms_iou=base.nms_iou,
        )

    def relative_to_root(self, path: Path) -> str:
        return self.ingest.relative_to_root(path)

    def resolved(self) -> dict[str, Any]:
        """The merged config for manifests; paths stay relative to DATA_ROOT."""
        return {**self.raw, "profile": self.profile}

    def _check_model(self, model: str) -> None:
        if model not in self.raw["models"]:
            raise ConfigError(f"unknown model {model!r}; expected one of {', '.join(MODELS)}")


def load_detect_config(
    config_path: str | Path = DEFAULT_DETECT_CONFIG, profile: str = "local"
) -> DetectConfig:
    """Load the base detect config merged with the `<profile>.yaml` next to it."""
    config_path = Path(config_path)
    raw = _read_yaml(config_path)
    profile_path = config_path.with_name(f"{profile}.yaml")
    if profile_path == config_path:
        raise ConfigError(f"profile {profile!r} collides with the base config file name")
    raw = _merge(raw, _read_yaml(profile_path))

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ConfigError(f"missing config keys: {', '.join(missing)}")
    for model, entry in raw["models"].items():
        found = entry.get("candidates") or []
        if not found:
            raise ConfigError(f"model {model!r} has no candidates")
        if len(found) > MAX_CANDIDATES:
            raise ConfigError(
                f"model {model!r} has {len(found)} candidates; at most {MAX_CANDIDATES} allowed"
            )

    repo_root = _find_repo_root(config_path)
    ingest = load_config(repo_root / raw["ingest_config"], raw["ingest_profile"])
    return DetectConfig(raw=raw, ingest=ingest, profile=profile)


def check_vision_extra() -> None:
    """Raise a ConfigError with an install hint when the `vision` extra is not installed."""
    missing = [m for m in VISION_MODULES if importlib.util.find_spec(m) is None]
    if missing:
        raise ConfigError(
            f"missing packages for the detect commands: {', '.join(missing)}; "
            "install them with `uv sync --extra vision`"
        )

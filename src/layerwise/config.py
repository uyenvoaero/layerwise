import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from layerwise.errors import ConfigError

DEFAULT_CONFIG = "configs/ingest/base.yaml"
REQUIRED_KEYS = (
    "dataset",
    "source_dir",
    "processed_dir",
    "class_mapping",
    "known_builds",
    "expected_size",
    "subset",
    "tracking",
    "failure_threshold",
)


@dataclass(frozen=True)
class IngestConfig:
    """Resolved ingest configuration. `raw` keeps paths relative, as written in the YAML."""

    raw: dict[str, Any]
    data_root: Path
    repo_root: Path

    @property
    def dataset(self) -> str:
        return self.raw["dataset"]

    @property
    def source_dir(self) -> Path:
        return self.data_root / self.raw["source_dir"]

    @property
    def processed_dir(self) -> Path:
        return self.data_root / self.raw["processed_dir"]

    @property
    def class_mapping_path(self) -> Path:
        return self.repo_root / self.raw["class_mapping"]

    @property
    def fixture_dir(self) -> Path:
        return self.repo_root / self.raw.get("fixture_dir", f"tests/fixtures/{self.dataset}")

    @property
    def known_builds(self) -> list[str]:
        return list(self.raw["known_builds"])

    @property
    def expected_size(self) -> dict[str, tuple[int, int]]:
        return {k: (int(v[0]), int(v[1])) for k, v in self.raw["expected_size"].items()}

    @property
    def selected_builds(self) -> list[str]:
        builds = self.raw["subset"].get("builds", "all")
        return self.known_builds if builds == "all" else list(builds)

    @property
    def selected_modalities(self) -> list[str]:
        return list(self.raw["subset"].get("modalities", ["PB", "OT"]))

    @property
    def max_layers_per_build(self) -> int | None:
        return self.raw["subset"].get("max_layers_per_build")

    @property
    def mlflow_enabled(self) -> bool:
        return bool(self.raw["tracking"].get("mlflow", False))

    @property
    def mlflow_dir(self) -> Path:
        return self.data_root / self.raw["tracking"].get("dir", "mlruns")

    @property
    def failure_threshold(self) -> float:
        return float(self.raw["failure_threshold"])

    @property
    def seed(self) -> int | None:
        return self.raw.get("seed")

    def relative_to_root(self, path: Path) -> str:
        return path.relative_to(self.data_root).as_posix()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _find_repo_root(config_path: Path) -> Path:
    for parent in config_path.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(config_path: str | Path = DEFAULT_CONFIG, profile: str = "local") -> IngestConfig:
    """Load the base config merged with the `<profile>.yaml` next to it."""
    config_path = Path(config_path)
    raw = _read_yaml(config_path)
    profile_path = config_path.with_name(f"{profile}.yaml")
    if profile_path == config_path:
        raise ConfigError(f"profile {profile!r} collides with the base config file name")
    raw = _merge(raw, _read_yaml(profile_path))

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ConfigError(f"missing config keys: {', '.join(missing)}")

    data_root_env = os.environ.get("DATA_ROOT")
    if not data_root_env:
        raise ConfigError("DATA_ROOT environment variable is not set")
    data_root = Path(data_root_env)
    if not data_root.is_dir():
        raise ConfigError(f"DATA_ROOT is not a directory: {data_root}")

    return IngestConfig(raw=raw, data_root=data_root, repo_root=_find_repo_root(config_path))

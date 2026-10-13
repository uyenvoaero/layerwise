from dataclasses import dataclass
from pathlib import Path

import yaml

from layerwise.errors import ConfigError, UnmappedClassError


@dataclass(frozen=True)
class ClassMapping:
    dataset: str
    version: str
    classes: dict[str, tuple[str, str | None]]  # source_class -> (phase_category, fine_class)

    def lookup(self, source_class: str) -> tuple[str, str | None]:
        try:
            return self.classes[source_class]
        except KeyError:
            raise UnmappedClassError(
                f"no class mapping entry for dataset {self.dataset!r}, class {source_class!r}"
            ) from None


def load_mapping(path: Path) -> ClassMapping:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ConfigError(f"class mapping file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict) or not all(k in data for k in ("version", "dataset", "classes")):
        raise ConfigError(f"class mapping {path} needs keys: version, dataset, classes")

    classes: dict[str, tuple[str, str | None]] = {}
    for source_class, entry in data["classes"].items():
        fine_class = entry.get("fine_class")
        if fine_class == "porosity":
            raise ConfigError(
                f"class mapping {data['dataset']!r}, class {source_class!r}: "
                "fine_class 'porosity' is not allowed for in-situ imaging labels"
            )
        classes[str(source_class)] = (entry["phase_category"], fine_class)
    return ClassMapping(data["dataset"], str(data["version"]), classes)

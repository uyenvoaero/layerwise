import hashlib
import json
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def code_version() -> str:
    """Package version (not a git hash) so reruns stay byte-identical."""
    try:
        return version("layerwise")
    except PackageNotFoundError:
        return "0+unknown"


def input_hash(data_root: Path, image_paths: Iterable[Path], xml_paths: Iterable[Path]) -> str:
    """Hash of sorted relative path + size for images and SHA-256 for XML files."""
    lines = []
    for path in image_paths:
        lines.append(f"img\t{path.relative_to(data_root).as_posix()}\t{path.stat().st_size}")
    for path in xml_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"xml\t{path.relative_to(data_root).as_posix()}\t{digest}")
    lines.sort()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def dumps_json(data: Any) -> str:
    """Deterministic JSON: sorted keys, two-space indent, trailing newline, no timestamps."""
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def config_hash(resolved_config: dict[str, Any]) -> str:
    return hashlib.sha256(dumps_json(resolved_config).encode("utf-8")).hexdigest()


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps_json(data))


def build_manifest(
    *,
    dataset: str,
    source: dict[str, str],
    adapter_version: str,
    input_hash_value: str,
    config: dict[str, Any],
    mapping_version: str,
    seed: int | None,
    outputs: dict[str, int],
    metrics: dict[str, int],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source": source,
        "code_version": code_version(),
        "adapter_version": adapter_version,
        "input_hash": input_hash_value,
        "config_hash": config_hash(config),
        "config": config,
        "mapping_version": mapping_version,
        "seed": seed,
        "outputs": outputs,
        "metrics": metrics,
    }

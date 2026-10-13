import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from layerwise.detect.config import DetectConfig
from layerwise.manifest import code_version, config_hash, write_json

LABEL_SOURCE = "aalto_manual_pb_boxes"
LIBRARIES = ("torch", "torchvision", "ultralytics", "opencv-python-headless")


@dataclass(frozen=True)
class JobKey:
    """Identity of one unit of work: (model, outer_fold, kind, inner_fold, candidate_id)."""

    model: str
    outer_fold: int
    kind: str  # inner | final | heldout
    inner_fold: int | None
    candidate_id: str

    def relative_dir(self) -> str:
        if self.kind == "inner":
            return f"inner_{self.inner_fold}/{self.candidate_id}"
        if self.kind == "final":
            return "final"
        return "."

    def label(self) -> str:
        parts = [self.model, f"outer_{self.outer_fold}", self.kind]
        if self.inner_fold is not None:
            parts.append(f"inner_{self.inner_fold}")
        parts.append(self.candidate_id)
        return "/".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "outer_fold": self.outer_fold,
            "kind": self.kind,
            "inner_fold": self.inner_fold,
            "candidate_id": self.candidate_id,
        }


def run_root(cfg: DetectConfig, model: str, outer_fold: int, repro: bool = False) -> Path:
    root = cfg.runs_dir / model / f"outer_{outer_fold}"
    return root / "repro" if repro else root


def job_dir(cfg: DetectConfig, key: JobKey, repro: bool = False) -> Path:
    return run_root(cfg, key.model, key.outer_fold, repro) / key.relative_dir()


def staging_dir(cfg: DetectConfig, key: JobKey, repro: bool = False) -> Path:
    """Where the job's tiles, label files and dataset YAML go; rebuildable, never committed."""
    name = key.relative_dir().replace("/", "_")
    base = cfg.staging_dir / key.model / f"outer_{key.outer_fold}"
    return (base / "repro" if repro else base) / name


def is_done(directory: Path, prefix: str = "") -> bool:
    return (directory / f"{prefix}result.json").exists()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def subset_hash(cfg: DetectConfig, key: JobKey, image_ids: list[str]) -> str:
    """Ingest input hash + fold definition + image IDs; identical inputs give identical hashes."""
    manifest_path = cfg.ingest.processed_dir / "manifest.json"
    ingest_hash = read_json(manifest_path)["input_hash"] if manifest_path.exists() else ""
    text = "\n".join([ingest_hash, json.dumps(key.as_dict(), sort_keys=True), *sorted(image_ids)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in LIBRARIES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def finish_job(
    cfg: DetectConfig,
    key: JobKey,
    directory: Path,
    *,
    result: dict[str, Any],
    image_ids: list[str],
    train_builds: list[str],
    validation_builds: list[str],
    test_builds: list[str],
    device: dict[str, Any],
    resolved_config: dict[str, Any],
    tracking_run_id: str | None,
    started_at: str,
    metrics: dict[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Write manifest.json, timing.json and, last, result.json (the done marker).

    `prefix` names the files of the held-out stage, which shares its directory with the fold.
    """
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "job": key.as_dict(),
        "data_subset_hash": subset_hash(cfg, key, image_ids),
        "config": resolved_config,
        "config_hash": config_hash(resolved_config),
        "code_version": code_version(),
        "seed": cfg.seed,
        "device": device,
        "libraries": library_versions(),
        "train_builds": sorted(train_builds),
        "validation_builds": sorted(validation_builds),
        "test_builds": sorted(test_builds),
        "metrics": metrics if metrics is not None else result,
        "label_source": LABEL_SOURCE,
        "tracking_run_id": tracking_run_id,
    }
    write_json(directory / f"{prefix}manifest.json", manifest)
    write_json(directory / f"{prefix}timing.json", {"started_at": started_at, "finished_at": now()})
    write_json(directory / f"{prefix}result.json", result)
    return manifest

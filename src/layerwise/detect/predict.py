"""Run a trained model (or the baseline) over layers and write the common defect representation."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layerwise.detect import representation
from layerwise.detect.config import DetectConfig
from layerwise.detect.data import load_fold_plan, pb_images, read_gray
from layerwise.detect.jobs import read_json, run_root
from layerwise.detect.models.base import Detection, predict_image
from layerwise.detect.protocol import DetectorFactory
from layerwise.errors import ConfigError, UnreadableImagesError


@dataclass
class PredictResult:
    directory: Path
    run: str
    threshold: float
    layers: int
    detections: int
    unreadable: list[str]


def parse_run(run: str) -> tuple[str, int]:
    """`<model>/outer_<k>` -> (model, k)."""
    parts = run.split("/")
    if len(parts) != 2 or not parts[1].startswith("outer_") or not parts[1][6:].isdigit():
        raise ConfigError(f"run must look like <model>/outer_<k>, got {run!r}")
    return parts[0], int(parts[1][6:])


def load_run(cfg: DetectConfig, run: str) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    """Model name, outer fold, selection and final result of a finished run."""
    model, k = parse_run(run)
    root = run_root(cfg, model, k)
    if not (root / "selection.json").exists() or not (root / "final" / "result.json").exists():
        raise ConfigError(f"run {run} not found under {cfg.runs_dir}; train it first")
    return model, k, read_json(root / "selection.json"), read_json(root / "final" / "result.json")


def run_predict(
    cfg: DetectConfig,
    run: str,
    factory: DetectorFactory,
    *,
    build: str | None = None,
    limit: int | None = None,
    threshold: float | None = None,
) -> PredictResult:
    model, k, selection, final = load_run(cfg, run)
    fold = next((f for f in load_fold_plan(cfg).outer_folds if f.outer_fold == k), None)
    if fold is None:
        raise ConfigError(f"outer fold {k} is not in the fold table")
    build = build or fold.test_build
    images = pb_images(cfg, [build])
    if not images:
        raise ConfigError(f"no PB images found for build {build!r}")
    if limit is not None:
        images = images[:limit]
    threshold = selection["threshold"] if threshold is None else float(threshold)

    candidate = next(c for c in cfg.candidates(model) if c.id == selection["winner"])
    detector = factory(model, candidate)
    root = run_root(cfg, model, k)
    detector.load(root / "final" / final["weights"] if final["weights"] else None)
    tp = cfg.tiling_for(candidate)

    layers, unreadable = [], []
    for image in images:
        try:
            gray = read_gray(image.path)
        except (OSError, ValueError):
            unreadable.append(image.image_id)
            continue
        boxes, scores = predict_image(gray, detector, tp)
        layers.append(
            representation.LayerOutput(
                image.image_id,
                image.build_id,
                image.layer_index,
                image.size_flag,
                gray.shape[1],
                gray.shape[0],
                [
                    Detection(tuple(float(v) for v in box), float(score))
                    for box, score in zip(boxes, scores, strict=True)
                ],
            )
        )
    if unreadable and len(unreadable) / len(images) > cfg.ingest.failure_threshold:
        raise UnreadableImagesError(
            f"{len(unreadable)} of {len(images)} layers are unreadable "
            f"(failure_threshold {cfg.ingest.failure_threshold})"
        )

    directory = cfg.predictions_dir / model / f"outer_{k}"
    if directory.exists():
        shutil.rmtree(directory)
    _, detection_rows = representation.write_outputs(
        directory,
        cfg.data_root,
        layers,
        model_id=f"{model}/{selection['winner']}",
        manifest_path=cfg.relative_to_root(root / "final" / "manifest.json"),
        threshold=threshold,
    )
    return PredictResult(directory, run, threshold, len(layers), len(detection_rows), unreadable)

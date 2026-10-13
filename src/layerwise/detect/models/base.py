"""Detector interface shared by the baseline, YOLO, Faster R-CNN and (later) segmentation models.

The protocol, metrics, prediction writer and latency benchmark only use this interface. The
Phase B segmentation model implements the same interface later: `predict_tiles` then returns boxes
from connected components and the defect map is written from its per-pixel confidence (see
specs/002-powder-bed-detection/contracts/defect-representation.md).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from layerwise.detect import tiling
from layerwise.detect.config import Candidate, TilingParams

SOURCE_CLASS = "defects"
PHASE_CATEGORY = "recoating_phase_anomaly"


@dataclass(frozen=True)
class Detection:
    """One predicted defect, box in image pixels."""

    box: tuple[float, float, float, float]
    confidence: float
    source_class: str = SOURCE_CLASS
    phase_category: str = PHASE_CATEGORY


@dataclass(frozen=True)
class TileEntry:
    """A training or validation tile: a crop of a source image with boxes in tile coordinates."""

    image_id: str
    image_path: Path
    tile: tuple[int, int, int, int]
    boxes: np.ndarray  # (N, 4) in tile coordinates; empty for a negative tile


@dataclass(frozen=True)
class FitJob:
    """Everything a detector needs to train one job."""

    directory: Path  # job directory (weights/ lives here)
    staging_dir: Path
    candidate: Candidate
    tiling: TilingParams
    epochs: int
    patience: int
    seed: int
    train: dict[str, Any]  # batch_size, workers, amp, device
    options: dict[str, Any] = field(default_factory=dict)  # model options from the config
    weights_dir: Path | None = None


@dataclass(frozen=True)
class FitResult:
    """`best_epoch` is 1-based; `history` has one dict per epoch."""

    best_epoch: int
    epochs_run: int
    best_weights: Path
    last_weights: Path
    history: list[dict[str, Any]] = field(default_factory=list)


class Detector(Protocol):
    name: str
    trainable: bool

    def fit(
        self,
        job: FitJob,
        train_tiles: list[TileEntry],
        validation_tiles: list[TileEntry] | None,
        resume: bool,
    ) -> FitResult: ...

    def load(self, path: Path | None) -> None:
        """Load trained weights; a non-learned model ignores `path`."""

    def predict_tiles(self, tiles: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Gray tile arrays -> per tile (boxes (N, 4) in tile pixels, confidences (N,))."""

    def info(self) -> dict[str, Any]:
        """model, runtime, precision, device, device_name, tile_batch_size, ..."""


def predict_image(
    image: np.ndarray, detector: Detector, params: TilingParams
) -> tuple[np.ndarray, np.ndarray]:
    """Tile a gray image, predict every tile, shift to image pixels and merge with NMS."""
    height, width = image.shape[:2]
    grid = tiling.tile_grid(width, height, params.tile_size, params.overlap)
    results = detector.predict_tiles([tiling.crop(image, t) for t in grid])
    shifted = []
    for tile, (boxes, scores) in zip(grid, results, strict=True):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        shifted.append((boxes + np.array([tile[0], tile[1], tile[0], tile[1]]), scores))
    boxes, scores = tiling.stitch(shifted, params.nms_iou)
    if len(boxes):
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes, scores

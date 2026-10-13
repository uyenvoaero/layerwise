"""Non-learned local-contrast baseline implementing `Detector`.

Background = large median blur; the absolute difference is divided by a robust local noise
estimate (median of the difference, scaled to a standard deviation), thresholded at a configured
multiple, opened, grouped into connected components and filtered by area. Confidence is the mean
normalised contrast of a component, scaled to [0, 1] by the configured saturation.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from layerwise.detect.config import Candidate, DetectConfig
from layerwise.detect.models.base import FitJob, FitResult, TileEntry

MAD_TO_SIGMA = 1.4826  # median absolute deviation -> standard deviation for Gaussian noise
NOISE_FLOOR = 1.0  # gray levels; keeps the ratio finite on perfectly flat tiles


def _odd(size: int) -> int:
    return size if size % 2 == 1 else size + 1


class ContrastDetector:
    name = "contrast"
    trainable = False

    def __init__(self, cfg: DetectConfig, candidate: Candidate):
        options = cfg.model_options("contrast")
        self.candidate = candidate
        self.blur = _odd(int(options["background_blur"]))
        self.threshold_multiple = float(options["threshold_multiple"])
        self.min_area = int(options["min_area"])
        self.opening = int(options["opening_size"])
        self.saturation = float(options["confidence_saturation"])

    def fit(
        self,
        job: FitJob,
        train_tiles: list[TileEntry],
        validation_tiles: list[TileEntry] | None,
        resume: bool,
    ) -> FitResult:
        raise NotImplementedError("the contrast baseline is not trained")

    def load(self, path: Path | None) -> None:
        """Nothing to load: all parameters come from the config."""

    def predict_tiles(self, tiles: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self._detect(tile) for tile in tiles]

    def _detect(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tile = np.ascontiguousarray(tile, dtype=np.uint8)
        background = cv2.medianBlur(tile, self.blur)
        diff = cv2.absdiff(tile, background)
        noise = np.maximum(
            MAD_TO_SIGMA * cv2.medianBlur(diff, self.blur).astype(np.float32), NOISE_FLOOR
        )
        score = diff.astype(np.float32) / noise
        mask = (score >= self.threshold_multiple).astype(np.uint8)
        if self.opening > 1:
            kernel = np.ones((self.opening, self.opening), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes, confidences = [], []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < self.min_area:
                continue
            mean_score = float(score[labels == label].mean())
            boxes.append([x, y, x + w, y + h])
            confidences.append(min(mean_score / self.saturation, 1.0))
        order = np.argsort(-np.array(confidences), kind="stable") if confidences else []
        return (
            np.array(boxes, dtype=float).reshape(-1, 4)[order],
            np.array(confidences, dtype=float)[order],
        )

    def info(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "runtime": "opencv",
            "precision": "fp32",
            "device": "cpu",
            "device_name": None,
            "tile_batch_size": 1,
        }

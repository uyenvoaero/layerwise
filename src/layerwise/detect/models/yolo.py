"""Ultralytics YOLOv8 wrapper implementing `Detector`.

Ultralytics is licensed under AGPL-3.0. It is used here as a library for internal research; its
terms apply if this project is ever distributed or served to others (see README).
"""

import contextlib
import csv
import json
import shutil
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from layerwise.detect.config import Candidate, DetectConfig
from layerwise.detect.models.base import FitJob, FitResult, TileEntry
from layerwise.detect.staging import write_yolo_dataset

FIT_FILE = "fit.json"


def resolve_device(requested: str) -> str:
    """`auto` picks CUDA when available; returns `cuda` or `cpu`."""
    import torch

    if requested in ("auto", "cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _ultralytics_device(device: str) -> str:
    return "0" if device == "cuda" else "cpu"


@contextlib.contextmanager
def _integrations_off():
    """Our protocol logs jobs to MLflow itself, so Ultralytics' own callbacks stay off."""
    with mock.patch("ultralytics.utils.callbacks.add_integration_callbacks", lambda instance: None):
        yield


class YoloDetector:
    name = "yolo"
    trainable = True

    def __init__(self, cfg: DetectConfig, candidate: Candidate):
        self.cfg = cfg
        self.candidate = candidate
        train = cfg.train
        self.device = resolve_device(str(train.get("device", "auto")))
        self.half = bool(train.get("amp", False)) and self.device == "cuda"
        self.tile_batch_size = int(train.get("batch_size", 16))
        self.conf_floor = cfg.eval_min_confidence
        self.model: Any = None

    # -- training ----------------------------------------------------------------------------

    def _weights_source(self, job: FitJob) -> str:
        name = self.candidate.weights or "yolov8n.yaml"
        if name.endswith(".yaml"):
            return name  # built from scratch, nothing to download
        path = (job.weights_dir or Path(".")) / name
        if not path.exists():
            from ultralytics.utils.downloads import attempt_download_asset

            path.parent.mkdir(parents=True, exist_ok=True)
            attempt_download_asset(str(path))
        return str(path)

    def fit(
        self,
        job: FitJob,
        train_tiles: list[TileEntry],
        validation_tiles: list[TileEntry] | None,
        resume: bool,
    ) -> FitResult:
        from ultralytics import YOLO

        directory = job.directory
        weights_dir = directory / "weights"
        fit_file = directory / FIT_FILE
        if resume and fit_file.exists() and (weights_dir / "last.pt").exists():
            return self._read_fit(directory)  # training finished earlier; only evaluation is left

        staged = write_yolo_dataset(job.staging_dir, train_tiles, validation_tiles)
        last = weights_dir / "last.pt"
        train = job.train
        common = {
            "epochs": job.epochs,
            "imgsz": job.tiling.tile_size,
            "batch": int(train["batch_size"]),
            "workers": int(train["workers"]),
            "device": _ultralytics_device(self.device),
            "amp": bool(train.get("amp", False)) and self.device == "cuda",
            "seed": job.seed,
            "deterministic": True,
            "patience": job.patience,
            "val": validation_tiles is not None,
            "plots": False,
            "verbose": False,
            "project": str(directory.parent),
            "name": directory.name,
            "exist_ok": True,
        }
        with _integrations_off():
            if resume and last.exists():
                YOLO(str(last)).train(resume=True)
            else:
                if directory.exists() and weights_dir.exists():
                    shutil.rmtree(weights_dir)
                YOLO(self._weights_source(job)).train(
                    data=str(staged.yaml_path), pretrained=False, **common
                )
        return self._write_fit(directory, validated=validation_tiles is not None)

    def _write_fit(self, directory: Path, validated: bool) -> FitResult:
        history = _read_results(directory / "results.csv")
        epochs_run = max(len(history), 1)
        best_epoch = _best_epoch(history) if validated and history else epochs_run
        (directory / FIT_FILE).write_text(
            json.dumps(
                {"best_epoch": best_epoch, "epochs_run": epochs_run, "history": history},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return self._read_fit(directory)

    @staticmethod
    def _read_fit(directory: Path) -> FitResult:
        data = json.loads((directory / FIT_FILE).read_text(encoding="utf-8"))
        weights = directory / "weights"
        best = weights / "best.pt"
        return FitResult(
            best_epoch=data["best_epoch"],
            epochs_run=data["epochs_run"],
            best_weights=best if best.exists() else weights / "last.pt",
            last_weights=weights / "last.pt",
            history=data["history"],
        )

    # -- inference ---------------------------------------------------------------------------

    def load(self, path: Path | None) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(path))

    def predict_tiles(self, tiles: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(tiles), self.tile_batch_size):
            chunk = tiles[start : start + self.tile_batch_size]
            batch = [np.repeat(t[..., None], 3, axis=2) for t in chunk]
            results = self.model.predict(
                source=batch,
                imgsz=self.candidate.tile_size,
                conf=self.conf_floor,
                device=_ultralytics_device(self.device),
                quantize="fp16" if self.half else "fp32",
                verbose=False,
            )
            for result in results:
                boxes = result.boxes
                out.append(
                    (boxes.xyxy.cpu().numpy().astype(float), boxes.conf.cpu().numpy().astype(float))
                )
        return out

    def info(self) -> dict[str, Any]:
        name = None
        if self.device == "cuda":
            import torch

            name = torch.cuda.get_device_name(0)
        return {
            "model": self.name,
            "runtime": "pytorch",
            "precision": "fp16" if self.half else "fp32",
            "device": self.device,
            "device_name": name,
            "tile_batch_size": self.tile_batch_size,
            "weights": self.candidate.weights,
        }


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    history = []
    for row in rows:
        clean = {k.strip(): v.strip() for k, v in row.items() if k}
        history.append({k: _number(v) for k, v in clean.items()})
    return history


def _number(text: str) -> float | str:
    try:
        return float(text)
    except ValueError:
        return text


def _best_epoch(history: list[dict[str, Any]]) -> int:
    """Epoch whose checkpoint became `best.pt` (fitness 0.1 mAP50 + 0.9 mAP50-95; ties go later)."""
    best_epoch, best_fitness = len(history), None
    for number, row in enumerate(history, start=1):
        fitness = 0.1 * float(row.get("metrics/mAP50(B)", 0.0)) + 0.9 * float(
            row.get("metrics/mAP50-95(B)", 0.0)
        )
        if best_fitness is None or fitness >= best_fitness:
            best_epoch, best_fitness = number, fitness
    return best_epoch

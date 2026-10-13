"""torchvision Faster R-CNN (ResNet-50 FPN v2) wrapper implementing `Detector`.

A small training loop on the same tile index as YOLO (crops are made in memory). A checkpoint with
optimizer state is written every epoch; `resume` continues from the latest one.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from layerwise.detect import metrics
from layerwise.detect.config import Candidate, DetectConfig
from layerwise.detect.data import read_gray
from layerwise.detect.models.base import FitJob, FitResult, TileEntry
from layerwise.detect.models.yolo import resolve_device

NUM_CLASSES = 2  # background + defects


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _to_tensor(tile: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.repeat(tile[None], 3, axis=0).astype(np.float32) / 255.0)


class FasterRcnnDetector:
    name = "frcnn"
    trainable = True

    def __init__(self, cfg: DetectConfig, candidate: Candidate, tile_size: int | None = None):
        self.cfg = cfg
        self.candidate = candidate
        self.tile_size = tile_size or candidate.tile_size
        train = cfg.train
        self.device = resolve_device(str(train.get("device", "auto")))
        self.amp = bool(train.get("amp", False)) and self.device == "cuda"
        self.tile_batch_size = int(train.get("batch_size", 8))
        self.options = cfg.model_options("frcnn")
        self.model: Any = None
        self._images: dict[Path, np.ndarray] = {}

    # -- model -------------------------------------------------------------------------------

    def new_model(self):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        pretrained = bool(self.options.get("pretrained", False))
        kwargs: dict[str, Any] = {
            "min_size": self.tile_size,
            "max_size": self.tile_size,
            "box_score_thresh": self.cfg.eval_min_confidence,
            "box_detections_per_img": 300,
        }
        if pretrained:
            model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT", **kwargs)
        else:
            model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kwargs)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        return model.to(self.device)

    @staticmethod
    def make_optimizer(model, options: dict[str, Any]):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=float(options.get("learning_rate", 1e-4)),
            weight_decay=float(options.get("weight_decay", 1e-4)),
        )

    # -- data --------------------------------------------------------------------------------

    def _tile_pixels(self, entry: TileEntry) -> np.ndarray:
        if entry.image_path not in self._images:
            self._images = {entry.image_path: read_gray(entry.image_path)}  # one image at a time
        x0, y0, x1, y1 = entry.tile
        return self._images[entry.image_path][y0:y1, x0:x1]

    def _batch(self, entries: list[TileEntry], augment: bool):
        images, targets = [], []
        for entry in entries:
            pixels, boxes = self._tile_pixels(entry), np.asarray(entry.boxes, float).reshape(-1, 4)
            if augment:
                pixels, boxes = _augment(pixels, boxes)
            images.append(_to_tensor(pixels).to(self.device))
            targets.append(
                {
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32)
                    .reshape(-1, 4)
                    .to(self.device),
                    "labels": torch.ones(len(boxes), dtype=torch.int64, device=self.device),
                }
            )
        return images, targets

    def train_step(self, model, optimizer, entries: list[TileEntry], scaler=None) -> float:
        """One optimisation step on a batch of tiles; returns the summed loss."""
        model.train()
        images, targets = self._batch(entries, augment=True)
        optimizer.zero_grad()
        with torch.autocast(self.device, enabled=self.amp):
            loss = sum(model(images, targets).values())
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return float(loss.detach().cpu())

    # -- training ----------------------------------------------------------------------------

    def fit(
        self,
        job: FitJob,
        train_tiles: list[TileEntry],
        validation_tiles: list[TileEntry] | None,
        resume: bool,
    ) -> FitResult:
        weights = job.directory / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        last, best = weights / "last.pt", weights / "best.pt"
        random.seed(job.seed)
        np.random.seed(job.seed)
        torch.manual_seed(job.seed)

        model = self.new_model()
        optimizer = self.make_optimizer(model, job.options)
        scaler = torch.amp.GradScaler(enabled=True) if self.amp else None
        state = {"epoch": 0, "history": [], "best_epoch": 0, "best_map": -1.0, "stopped": False}
        if resume and last.exists():
            checkpoint = load_checkpoint(last)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            state.update({k: checkpoint[k] for k in state})

        batch_size = int(job.train["batch_size"])
        while state["epoch"] < job.epochs and not state["stopped"]:
            order = list(range(len(train_tiles)))
            random.Random(job.seed + state["epoch"]).shuffle(order)
            losses = [
                self.train_step(
                    model, optimizer, [train_tiles[i] for i in order[s : s + batch_size]], scaler
                )
                for s in range(0, len(order), batch_size)
            ]
            state["epoch"] += 1
            row: dict[str, Any] = {
                "epoch": float(state["epoch"]),
                "train/loss": float(np.mean(losses)) if losses else 0.0,
            }
            improved = False
            if validation_tiles is not None:
                self.model = model
                score = self._validation_map(validation_tiles, batch_size)
                row["metrics/mAP50(B)"] = score
                if score > state["best_map"]:
                    state["best_map"], state["best_epoch"], improved = score, state["epoch"], True
                elif state["epoch"] - state["best_epoch"] >= job.patience:
                    state["stopped"] = True
            else:
                state["best_epoch"] = state["epoch"]
            state["history"].append(row)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                **state,
            }
            torch.save(checkpoint, last)
            if improved:
                torch.save({"model": model.state_dict(), **state}, best)
        if not best.exists():
            torch.save({"model": model.state_dict(), **state}, best)
        return FitResult(
            best_epoch=max(state["best_epoch"], 1),
            epochs_run=state["epoch"],
            best_weights=best,
            last_weights=last,
            history=state["history"],
        )

    def _validation_map(self, tiles: list[TileEntry], batch_size: int) -> float:
        self.model.eval()
        evals = []
        for start in range(0, len(tiles), batch_size):
            chunk = tiles[start : start + batch_size]
            pixels = [self._tile_pixels(t) for t in chunk]
            for entry, (boxes, scores) in zip(chunk, self.predict_tiles(pixels), strict=True):
                evals.append(metrics.ImageEval(entry.image_id, entry.boxes, boxes, scores))
        score = metrics.average_precision(metrics.match_images(evals, self.cfg.match_iou))
        return 0.0 if score is None else score

    # -- inference ---------------------------------------------------------------------------

    def load(self, path: Path | None) -> None:
        self.model = self.new_model()
        self.model.load_state_dict(load_checkpoint(path)["model"])
        self.model.eval()

    def predict_tiles(self, tiles: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        self.model.eval()
        out: list[tuple[np.ndarray, np.ndarray]] = []
        with torch.no_grad(), torch.autocast(self.device, enabled=self.amp):
            for start in range(0, len(tiles), self.tile_batch_size):
                chunk = tiles[start : start + self.tile_batch_size]
                batch = [_to_tensor(t).to(self.device) for t in chunk]
                for result in self.model(batch):
                    out.append(
                        (
                            result["boxes"].float().cpu().numpy().astype(float),
                            result["scores"].float().cpu().numpy().astype(float),
                        )
                    )
        return out

    def info(self) -> dict[str, Any]:
        name = torch.cuda.get_device_name(0) if self.device == "cuda" else None
        return {
            "model": self.name,
            "runtime": "pytorch",
            "precision": "fp16" if self.amp else "fp32",
            "device": self.device,
            "device_name": name,
            "tile_batch_size": self.tile_batch_size,
            "weights": "fasterrcnn_resnet50_fpn_v2",
        }


def _augment(pixels: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flips and brightness/contrast jitter (albumentations) with the boxes carried along."""
    import albumentations as A

    height, width = pixels.shape[:2]
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
    )
    out = transform(
        image=pixels,
        bboxes=boxes.clip(0, [width, height, width, height]).tolist(),
        labels=[1] * len(boxes),
    )
    new_boxes = np.array(out["bboxes"], dtype=float).reshape(-1, 4)
    return out["image"], new_boxes

"""Per-job tile sets and the Ultralytics dataset layout.

Tiles are written as new JPEG files under the staging directory; the source images are only read.
"""

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from layerwise.detect import tiling
from layerwise.detect.config import TilingParams
from layerwise.detect.data import LabeledImage
from layerwise.detect.models.base import TileEntry

SIGNATURE_FILE = "tiles.sha256"
JPEG_QUALITY = 95


@dataclass(frozen=True)
class StagedDataset:
    root: Path
    yaml_path: Path
    n_train: int
    n_validation: int


def build_tile_entries(
    images: list[LabeledImage], params: TilingParams, seed: int
) -> list[TileEntry]:
    """Positive tiles (at least one box with `min_box_visible` inside) plus seeded negatives.

    A tile that only touches a box without keeping it is neither positive nor negative, so a
    partly visible defect is never presented as defect-free.
    """
    positives: list[TileEntry] = []
    negatives: list[tuple[int, tiling.Tile]] = []
    for index, image in enumerate(images):
        for tile in tiling.tile_grid(image.width, image.height, params.tile_size, params.overlap):
            kept, _, _ = tiling.clip_boxes_to_tile(image.boxes, tile, params.min_box_visible)
            if len(kept):
                positives.append(TileEntry(image.image_id, image.path, tile, kept))
            elif not _touches_a_box(image.boxes, tile):
                negatives.append((index, tile))
    chosen = tiling.sample_negative_tiles(
        negatives, len(positives), params.negative_tile_ratio, seed
    )
    empty = np.zeros((0, 4))
    entries = positives + [
        TileEntry(images[i].image_id, images[i].path, tile, empty) for i, tile in chosen
    ]
    entries.sort(key=lambda e: (e.image_id, e.tile))
    return entries


def _touches_a_box(boxes: np.ndarray, tile: tiling.Tile) -> bool:
    x0, y0, x1, y1 = tile
    for bx0, by0, bx1, by1 in boxes:
        if min(bx1, x1) > max(bx0, x0) and min(by1, y1) > max(by0, y0):
            return True
    return False


def tile_name(entry: TileEntry) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", entry.image_id)
    return f"{safe}__x{entry.tile[0]}_y{entry.tile[1]}"


def _signature(train: list[TileEntry], validation: list[TileEntry] | None) -> str:
    def rows(entries):
        return [
            [tile_name(e), list(e.tile), np.asarray(e.boxes).round(3).tolist()] for e in entries
        ]

    payload = {"train": rows(train), "validation": None if validation is None else rows(validation)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def write_yolo_dataset(
    root: Path, train: list[TileEntry], validation: list[TileEntry] | None
) -> StagedDataset:
    """Write tile JPEGs, YOLO label files (class 0, normalised xywh) and the dataset YAML.

    Without a validation set `val` points at the training tiles; the final training turns
    validation off, so the entry only satisfies the Ultralytics dataset check.
    """
    root = Path(root)
    yaml_path = root / "dataset.yaml"
    signature = _signature(train, validation)
    marker = root / SIGNATURE_FILE
    if (
        yaml_path.exists()
        and marker.exists()
        and marker.read_text(encoding="utf-8") == signature
        and _all_files_present(root, train, validation)
    ):
        return StagedDataset(root, yaml_path, len(train), len(validation or []))
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _write_split(root, "train", train)
    if validation is not None:
        _write_split(root, "val", validation)
    config = {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val" if validation is not None else "images/train",
        "names": {0: "defects"},
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    marker.write_text(signature, encoding="utf-8")
    return StagedDataset(root, yaml_path, len(train), len(validation or []))


def _all_files_present(root: Path, train, validation) -> bool:
    for split, entries in (("train", train), ("val", validation or [])):
        for entry in entries:
            name = tile_name(entry)
            if not (root / "images" / split / f"{name}.jpg").exists():
                return False
            if not (root / "labels" / split / f"{name}.txt").exists():
                return False
    return True


def _write_split(root: Path, split: str, entries: list[TileEntry]) -> None:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    open_path: Path | None = None
    source: Image.Image | None = None
    for entry in entries:  # sorted by image, so each source file is opened once
        if entry.image_path != open_path:
            with Image.open(entry.image_path) as handle:
                source = handle.convert("L")
            open_path = entry.image_path
        x0, y0, x1, y1 = entry.tile
        name = tile_name(entry)
        source.crop((x0, y0, x1, y1)).save(
            images_dir / f"{name}.jpg", "JPEG", quality=JPEG_QUALITY
        )
        (labels_dir / f"{name}.txt").write_text(
            _label_text(entry.boxes, x1 - x0, y1 - y0), encoding="utf-8", newline="\n"
        )


def _label_text(boxes: np.ndarray, width: int, height: int) -> str:
    lines = []
    for bx0, by0, bx1, by1 in np.asarray(boxes, float).reshape(-1, 4):
        cx, cy = (bx0 + bx1) / 2 / width, (by0 + by1) / 2 / height
        w, h = (bx1 - bx0) / width, (by1 - by0) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")

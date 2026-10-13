from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from layerwise.detect.config import DetectConfig
from layerwise.errors import ConfigError, FoldOverlapError, UnreadableImagesError
from layerwise.ingest.checks import find_violations
from layerwise.ingest.sqlfiles import load_sql, open_tables

SQL_DIR = Path(__file__).parent / "sql"


@dataclass(frozen=True)
class LabeledImage:
    """A PB image with a label file; `boxes` is (N, 4) xmin, ymin, xmax, ymax in image pixels."""

    image_id: str
    build_id: str
    layer_index: int
    path: Path
    width: int
    height: int
    size_flag: bool
    boxes: np.ndarray


@dataclass(frozen=True)
class InnerFold:
    inner_fold: int
    train_builds: list[str]
    validation_builds: list[str]


@dataclass(frozen=True)
class OuterFold:
    outer_fold: int
    train_builds: list[str]
    test_build: str
    inner: list[InnerFold]


@dataclass(frozen=True)
class FoldPlan:
    outer_folds: list[OuterFold]

    @property
    def full_protocol(self) -> bool:
        """True when there are 3 outer folds with 2 inner folds each."""
        return len(self.outer_folds) >= 3 and all(len(f.inner) >= 2 for f in self.outer_folds)


def _folds_path(cfg: DetectConfig) -> Path:
    path = cfg.ingest.processed_dir / "folds.parquet"
    if not path.exists():
        raise ConfigError(f"{path} not found; run `layerwise ingest` first")
    return path


def load_fold_plan(cfg: DetectConfig) -> FoldPlan:
    """Read the fold table; the feature 001 overlap check runs first and stops on any violation."""
    rows = pq.read_table(_folds_path(cfg)).to_pylist()
    violations = find_violations(rows)
    if violations:
        raise FoldOverlapError("; ".join(v.message for v in violations))

    outer: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    inner: dict[tuple[int, int], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["level"] == "outer":
            outer[r["outer_fold"]][r["role"]].append(r["build_id"])
        else:
            inner[(r["outer_fold"], r["inner_fold"])][r["role"]].append(r["build_id"])
    folds = []
    for k in sorted(outer):
        roles = outer[k]
        inner_folds = [
            InnerFold(j, sorted(inner[(k, j)]["train"]), sorted(inner[(k, j)]["validation"]))
            for (ok, j) in sorted(inner)
            if ok == k
        ]
        folds.append(OuterFold(k, sorted(roles["train"]), roles["test"][0], inner_folds))
    return FoldPlan(folds)


def labeled_images(cfg: DetectConfig, builds: list[str]) -> list[LabeledImage]:
    """Labeled PB images of the given builds (never `unlabeled`), sorted by build and layer."""
    tables = ["images", "boxes"]
    con = open_tables(cfg.ingest.processed_dir, tables)
    try:
        image_rows = con.execute(load_sql("fold_data.sql", "labeled_images", SQL_DIR)).fetchall()
        box_rows = con.execute(load_sql("fold_data.sql", "labeled_boxes", SQL_DIR)).fetchall()
    finally:
        con.close()
    boxes: dict[str, list[list[int]]] = defaultdict(list)
    for image_id, x0, y0, x1, y1 in box_rows:
        boxes[image_id].append([x0, y0, x1, y1])
    wanted = set(builds)
    return [
        LabeledImage(
            image_id=image_id,
            build_id=build_id,
            layer_index=layer_index,
            path=cfg.data_root / path,
            width=width,
            height=height,
            size_flag=size_flag,
            boxes=np.array(boxes.get(image_id, []), dtype=float).reshape(-1, 4),
        )
        for image_id, build_id, layer_index, path, width, height, size_flag in image_rows
        if build_id in wanted
    ]


def pb_images(cfg: DetectConfig, builds: list[str]) -> list[LabeledImage]:
    """All PB layer images of the given builds (labeled or not), sorted by build and layer.

    `boxes` is empty: this is the image list for prediction, not for evaluation.
    """
    con = open_tables(cfg.ingest.processed_dir, ["images"])
    try:
        rows = con.execute(load_sql("fold_data.sql", "pb_images", SQL_DIR)).fetchall()
    finally:
        con.close()
    wanted = set(builds)
    return [
        LabeledImage(
            image_id=image_id,
            build_id=build_id,
            layer_index=layer_index,
            path=cfg.data_root / path,
            width=width,
            height=height,
            size_flag=size_flag,
            boxes=np.zeros((0, 4)),
        )
        for image_id, build_id, layer_index, path, width, height, size_flag in rows
        if build_id in wanted
    ]


def labeled_counts(cfg: DetectConfig) -> dict[str, dict[str, int]]:
    """Labeled PB images and boxes per build."""
    con = open_tables(cfg.ingest.processed_dir, ["images", "boxes"])
    try:
        rows = con.execute(load_sql("fold_data.sql", "labeled_counts", SQL_DIR)).fetchall()
    finally:
        con.close()
    return {build: {"images": images, "boxes": boxes} for build, images, boxes in rows}


def read_gray(path: Path) -> np.ndarray:
    """Decode an image to a gray uint8 array (H, W)."""
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"))


def filter_readable(cfg: DetectConfig, images: list[LabeledImage]) -> list[LabeledImage]:
    """Drop labeled images that cannot be decoded; stop when the failed share is too high."""
    readable = []
    for image in images:
        try:
            with Image.open(image.path) as handle:
                handle.load()
        except (OSError, ValueError):
            continue
        readable.append(image)
    failed = len(images) - len(readable)
    if images and failed / len(images) > cfg.ingest.failure_threshold:
        raise UnreadableImagesError(
            f"{failed} of {len(images)} labeled images are missing or unreadable "
            f"(failure_threshold {cfg.ingest.failure_threshold})"
        )
    return readable

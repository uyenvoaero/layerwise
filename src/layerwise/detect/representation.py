"""Common per-layer defect representation, format_version "1".

Tables and maps are written exactly as specified in
specs/002-powder-bed-detection/contracts/defect-representation.md. The schemas are defined here
once; the map of a box model is a deterministic function of the stored detections.
"""

import io
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from layerwise.detect.models.base import PHASE_CATEGORY, SOURCE_CLASS, Detection

FORMAT_VERSION = "1"
CATEGORIES = [PHASE_CATEGORY]  # fixed channel order; new phase categories are appended
STATUS_PROCESSED = "processed"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed so identical inputs give identical map files


def _schema(*fields: tuple[str, pa.DataType, bool]) -> pa.Schema:
    return pa.schema([pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in fields])


LAYERS_SCHEMA = _schema(
    ("image_id", pa.string(), False),
    ("build_id", pa.string(), False),
    ("layer_index", pa.int32(), False),
    ("size_flag", pa.bool_(), False),
    ("status", pa.string(), False),
    ("n_detections", pa.int32(), False),
    ("map_path", pa.string(), False),
    ("model_id", pa.string(), False),
    ("manifest_path", pa.string(), False),
    ("threshold", pa.float64(), False),
    ("format_version", pa.string(), False),
)

DETECTIONS_SCHEMA = _schema(
    ("detection_id", pa.string(), False),
    ("image_id", pa.string(), False),
    ("xmin", pa.int32(), False),
    ("ymin", pa.int32(), False),
    ("xmax", pa.int32(), False),
    ("ymax", pa.int32(), False),
    ("source_class", pa.string(), False),
    ("phase_category", pa.string(), False),
    ("confidence", pa.float32(), False),
)


@dataclass
class LayerOutput:
    """One processed layer: image identity and size plus its detections (image pixels)."""

    image_id: str
    build_id: str
    layer_index: int
    size_flag: bool
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)


def detection_rows(layer: LayerOutput, threshold: float) -> list[dict]:
    """Stored detections of a layer: integer boxes clipped to the image, confidence >= threshold,
    ids numbered from 0 in descending confidence."""
    kept = sorted(
        (d for d in layer.detections if d.confidence >= threshold),
        key=lambda d: -d.confidence,
    )
    rows = []
    for det in kept:
        xmin = max(math.floor(det.box[0]), 0)
        ymin = max(math.floor(det.box[1]), 0)
        xmax = min(math.ceil(det.box[2]), layer.width)
        ymax = min(math.ceil(det.box[3]), layer.height)
        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate after clipping
        rows.append(
            {
                "detection_id": f"{layer.image_id}#{len(rows)}",
                "image_id": layer.image_id,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "source_class": det.source_class,
                "phase_category": det.phase_category,
                "confidence": float(np.float32(min(max(det.confidence, 0.0), 1.0))),
            }
        )
    return rows


def rasterize(
    rows: list[dict], width: int, height: int, categories: list[str] = CATEGORIES
) -> np.ndarray:
    """Map (C, H, W) float16: per category the highest confidence of the covering boxes, else 0."""
    out = np.zeros((len(categories), height, width), dtype=np.float16)
    index = {name: i for i, name in enumerate(categories)}
    for row in rows:
        channel = out[index[row["phase_category"]]]
        region = channel[row["ymin"] : row["ymax"], row["xmin"] : row["xmax"]]
        np.maximum(region, np.float16(row["confidence"]), out=region)
    return out


def write_map(path: Path, defect_map: np.ndarray, categories: list[str] = CATEGORIES) -> None:
    """Compressed archive with `defect_map`, `categories` and `format_version`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "defect_map": defect_map,
        "categories": np.array(categories),
        "format_version": np.array(FORMAT_VERSION),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def read_map(path: Path) -> tuple[np.ndarray, list[str], str]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            archive["defect_map"],
            [str(c) for c in archive["categories"]],
            str(archive["format_version"]),
        )


def rebuild_map(
    detections: list[dict], image_id: str, width: int, height: int, categories=CATEGORIES
) -> np.ndarray:
    """Rebuild a layer's map from rows of detections.parquet."""
    rows = [r for r in detections if r["image_id"] == image_id]
    return rasterize(rows, width, height, categories)


def write_outputs(
    directory: Path,
    data_root: Path,
    layers: list[LayerOutput],
    *,
    model_id: str,
    manifest_path: str,
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Write layers.parquet, detections.parquet and maps/<image_id>.npz under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    layer_rows: list[dict] = []
    all_detections: list[dict] = []
    for layer in sorted(layers, key=lambda lay: lay.image_id):
        rows = detection_rows(layer, threshold)
        map_file = directory / "maps" / f"{layer.image_id}.npz"
        write_map(map_file, rasterize(rows, layer.width, layer.height))
        layer_rows.append(
            {
                "image_id": layer.image_id,
                "build_id": layer.build_id,
                "layer_index": layer.layer_index,
                "size_flag": layer.size_flag,
                "status": STATUS_PROCESSED,
                "n_detections": len(rows),
                "map_path": map_file.relative_to(data_root).as_posix(),
                "model_id": model_id,
                "manifest_path": manifest_path,
                "threshold": float(threshold),
                "format_version": FORMAT_VERSION,
            }
        )
        all_detections += rows
    pq.write_table(
        pa.Table.from_pylist(layer_rows, schema=LAYERS_SCHEMA),
        directory / "layers.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(all_detections, schema=DETECTIONS_SCHEMA),
        directory / "detections.parquet",
        compression="zstd",
    )
    return layer_rows, all_detections


__all__ = [
    "CATEGORIES",
    "DETECTIONS_SCHEMA",
    "FORMAT_VERSION",
    "LAYERS_SCHEMA",
    "SOURCE_CLASS",
    "LayerOutput",
    "detection_rows",
    "rasterize",
    "rebuild_map",
    "read_map",
    "write_map",
    "write_outputs",
]

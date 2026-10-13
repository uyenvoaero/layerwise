import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from layerwise.ingest.adapters.base import RawBox


@dataclass(frozen=True)
class ClippedBox:
    index: int  # position of the object in the XML (source order)
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    source_class: str
    clipped: bool


@dataclass(frozen=True)
class ClipResult:
    boxes: list[ClippedBox]
    ingested: int
    clipped: int
    dropped: int
    dropped_classes: list[str]


def read_voc(path: Path) -> list[RawBox]:
    """Parse the boxes of a Pascal VOC file. `<path>` and `<folder>` are ignored on purpose."""
    try:
        root = ET.parse(path).getroot()
        boxes = []
        for obj in root.iter("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                raise ValueError("object without bndbox")
            name = (obj.findtext("name") or "").strip()
            if not name:
                raise ValueError("object without name")
            coords = [float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            boxes.append(RawBox(*coords, source_class=name))
        return boxes
    except (ET.ParseError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed VOC file {path.name}: {exc}") from exc


def clip_boxes(boxes: list[RawBox], width: int, height: int) -> ClipResult:
    """Clip boxes to the image; boxes with zero area afterwards are dropped and counted."""
    kept: list[ClippedBox] = []
    dropped_classes: list[str] = []
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = round(box.xmin), round(box.ymin), round(box.xmax), round(box.ymax)
        cx0, cy0 = max(0, min(x0, width)), max(0, min(y0, height))
        cx1, cy1 = max(0, min(x1, width)), max(0, min(y1, height))
        if cx1 <= cx0 or cy1 <= cy0:
            dropped_classes.append(box.source_class)
            continue
        changed = (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1)
        kept.append(ClippedBox(index, cx0, cy0, cx1, cy1, box.source_class, changed))
    return ClipResult(
        boxes=kept,
        ingested=len(kept),
        clipped=sum(b.clipped for b in kept),
        dropped=len(dropped_classes),
        dropped_classes=dropped_classes,
    )

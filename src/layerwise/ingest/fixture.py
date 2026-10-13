import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

from layerwise.config import IngestConfig
from layerwise.ingest.adapters.aalto import AaltoAdapter
from layerwise.ingest.adapters.base import ImageRecord
from layerwise.ingest.voc import clip_boxes

LONGEST_SIDE = 320
IMAGES_PER_BUILD_AND_MODALITY = 20
MAX_TOTAL_BYTES = 5 * 1024 * 1024
JPEG_QUALITY = 85


class FixtureError(Exception):
    """The fixture cannot be built within its rules. Exit code 1."""


@dataclass(frozen=True)
class FixtureResult:
    images: int
    total_bytes: int


def _voc_xml(filename: str, width: int, height: int, boxes: list[tuple]) -> str:
    objects = "".join(
        f"\n  <object>\n    <name>{escape(name)}</name>\n    <bndbox>\n"
        f"      <xmin>{x0}</xmin>\n      <ymin>{y0}</ymin>\n"
        f"      <xmax>{x1}</xmax>\n      <ymax>{y1}</ymax>\n    </bndbox>\n  </object>"
        for name, x0, y0, x1, y1 in boxes
    )
    return (
        "<annotation>\n"
        f"  <filename>{escape(filename)}</filename>\n"
        f"  <size>\n    <width>{width}</width>\n    <height>{height}</height>\n"
        "    <depth>1</depth>\n  </size>"
        f"{objects}\n</annotation>\n"
    )


def _rescale(box, sx: float, sy: float, width: int, height: int) -> tuple:
    """Scale a clipped box to the downscaled image; keep at least one pixel of area."""
    x0 = max(0, min(round(box.xmin * sx), width - 1))
    y0 = max(0, min(round(box.ymin * sy), height - 1))
    x1 = max(x0 + 1, min(round(box.xmax * sx), width))
    y1 = max(y0 + 1, min(round(box.ymax * sy), height))
    return (box.source_class, x0, y0, x1, y1)


def _attribution(cfg: IngestConfig, info: dict[str, str]) -> str:
    return (
        "# Attribution\n\n"
        f"These images are downscaled derivatives (longest side {LONGEST_SIDE} px, boxes "
        "rescaled) of the following dataset, used only as a small test fixture.\n\n"
        f"- Source: {info['name']}\n"
        f"- DOI: {info['doi']}\n"
        f"- License: {info['license']}\n"
        f"- Content: per build, the first {IMAGES_PER_BUILD_AND_MODALITY} labeled powder bed (PB) "
        f"and {IMAGES_PER_BUILD_AND_MODALITY} labeled optical tomography (OT) images "
        f"({len(cfg.known_builds)} builds)\n"
        "- Regenerate with `uv run layerwise make-fixture`.\n"
    )


def make_fixture(cfg: IngestConfig) -> FixtureResult:
    """Write the committed smoke-test fixture from the full dataset into `cfg.fixture_dir`."""
    adapter = AaltoAdapter(cfg)
    records = list(adapter.discover())

    selected: list[tuple[ImageRecord, list]] = []
    for build in cfg.known_builds:
        for modality in ("PB", "OT"):
            candidates = sorted(
                (r for r in records if r.build_id == build and r.modality == modality),
                key=lambda r: (r.layer_index, r.image_id),
            )
            picked = []
            for rec in candidates:
                boxes = adapter.labels(rec)
                if boxes:
                    picked.append((rec, boxes))
                if len(picked) == IMAGES_PER_BUILD_AND_MODALITY:
                    break
            if len(picked) < IMAGES_PER_BUILD_AND_MODALITY:
                raise FixtureError(
                    f"build {build} has only {len(picked)} labeled {modality} images, "
                    f"need {IMAGES_PER_BUILD_AND_MODALITY}"
                )
            selected.extend(picked)

    out = cfg.fixture_dir
    for folder in ("PB", "PB_label", "OT", "OT_label"):
        shutil.rmtree(out / folder, ignore_errors=True)
        (out / folder).mkdir(parents=True)

    for rec, raw_boxes in selected:
        with Image.open(rec.path) as img:
            width, height = img.size
            scale = LONGEST_SIDE / max(width, height)
            new_w, new_h = round(width * scale), round(height * scale)
            small = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            small.save(out / rec.modality / f"{rec.image_id}.jpg", "JPEG", quality=JPEG_QUALITY)
        clipped = clip_boxes(raw_boxes, width, height).boxes
        boxes = [_rescale(b, new_w / width, new_h / height, new_w, new_h) for b in clipped]
        xml = _voc_xml(f"{rec.image_id}.jpg", new_w, new_h, boxes)
        (out / f"{rec.modality}_label" / f"{rec.image_id}.xml").write_text(xml, encoding="utf-8")

    (out / "ATTRIBUTION.md").write_text(_attribution(cfg, adapter.info()), encoding="utf-8")
    total = sum(p.stat().st_size for p in Path(out).rglob("*") if p.is_file())
    if total > MAX_TOTAL_BYTES:
        raise FixtureError(f"fixture is {total} bytes, above the {MAX_TOTAL_BYTES} byte limit")
    return FixtureResult(images=len(selected), total_bytes=total)

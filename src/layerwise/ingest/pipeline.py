import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from layerwise.config import IngestConfig
from layerwise.errors import ConfigError, InputErrorsExceededError
from layerwise.ingest import report as report_mod
from layerwise.ingest import tables
from layerwise.ingest.adapters.aalto import AaltoAdapter
from layerwise.ingest.adapters.base import ImageRecord
from layerwise.ingest.folds import fold_notes, make_folds
from layerwise.ingest.mapping import ClassMapping, load_mapping
from layerwise.ingest.pairing import pair_layers
from layerwise.ingest.sqlfiles import open_tables
from layerwise.ingest.tables import write_table
from layerwise.ingest.voc import clip_boxes
from layerwise.manifest import build_manifest, code_version, config_hash, input_hash, write_json
from layerwise.tracking import log_run

logger = logging.getLogger(__name__)

TABLE_NAMES = ("images", "boxes", "class_mapping", "layer_pairs", "folds")
OUTPUT_FILES = (*(f"{name}.parquet" for name in TABLE_NAMES), "manifest.json", "ingest_report.json")


@dataclass
class IngestOutcome:
    status: str  # "written", "up_to_date" or "dry_run"
    manifest: dict[str, Any]
    report: dict[str, Any]


def select_records(cfg: IngestConfig, records: list[ImageRecord]) -> list[ImageRecord]:
    """Apply the config subset: builds, modalities and the first N layers per build and modality."""
    builds, modalities = set(cfg.selected_builds), set(cfg.selected_modalities)
    chosen = sorted(
        (r for r in records if r.build_id in builds and r.modality in modalities),
        key=lambda r: (r.build_id, r.modality, r.layer_index, r.image_id),
    )
    limit = cfg.max_layers_per_build
    if limit is None:
        return chosen
    kept, layers_seen = [], {}
    for record in chosen:
        seen = layers_seen.setdefault((record.build_id, record.modality), set())
        if record.layer_index in seen or len(seen) < limit:
            seen.add(record.layer_index)
            kept.append(record)
    return kept


def _is_up_to_date(cfg: IngestConfig, in_hash: str, cfg_hash: str, mapping_version: str) -> bool:
    manifest_path = cfg.processed_dir / "manifest.json"
    if not all((cfg.processed_dir / name).exists() for name in OUTPUT_FILES):
        return False
    old = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        old.get("input_hash") == in_hash
        and old.get("config_hash") == cfg_hash
        and old.get("mapping_version") == mapping_version
        and old.get("code_version") == code_version()
    )


def _swap(tmp: Path, target: Path) -> None:
    """Replace `target` with `tmp` so an interrupted run never leaves half-written output."""
    old = target.with_name(f".{target.name}.old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    tmp.rename(target)
    shutil.rmtree(old, ignore_errors=True)


def run_ingest(cfg: IngestConfig, *, force: bool = False, dry_run: bool = False) -> IngestOutcome:
    mapping = load_mapping(cfg.class_mapping_path)
    if mapping.dataset != cfg.dataset:
        raise ConfigError(
            f"class mapping is for dataset {mapping.dataset!r}, config is for {cfg.dataset!r}"
        )
    adapter = AaltoAdapter(cfg)
    records = select_records(cfg, list(adapter.discover()))
    image_files, xml_files = adapter.source_files()
    in_hash = input_hash(cfg.data_root, image_files, xml_files)
    cfg_hash = config_hash(cfg.raw)

    if not force and not dry_run and _is_up_to_date(cfg, in_hash, cfg_hash, mapping.version):
        return IngestOutcome(
            "up_to_date",
            json.loads((cfg.processed_dir / "manifest.json").read_text(encoding="utf-8")),
            json.loads((cfg.processed_dir / "ingest_report.json").read_text(encoding="utf-8")),
        )

    image_rows, box_rows, failed, dropped, n_files = _scan_labels(cfg, adapter, records, mapping)
    fraction = len(failed) / n_files if n_files else 0.0
    if fraction > cfg.failure_threshold:
        raise InputErrorsExceededError(
            f"{len(failed)} of {n_files} input files failed ({fraction:.1%}), above the "
            f"failure_threshold of {cfg.failure_threshold:.1%}"
        )

    target = cfg.processed_dir
    tmp = target.with_name(f".{target.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        outcome = _write_outputs(
            cfg, adapter, mapping, tmp, in_hash, image_rows, box_rows, failed, dropped
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if dry_run:
        shutil.rmtree(tmp, ignore_errors=True)
        outcome.status = "dry_run"
        return outcome
    _swap(tmp, target)
    try:
        log_run(cfg, outcome.manifest)
    except Exception:  # tracking must not undo a finished ingest
        logger.warning("MLflow logging failed; the ingest output is written", exc_info=True)
    return outcome


def _scan_labels(cfg, adapter, records, mapping: ClassMapping):
    """Read image headers and label files; returns table rows, failures and drop counts."""
    image_rows: list[dict] = []
    box_rows: list[dict] = []
    failed: list[dict] = []
    dropped: Counter = Counter()
    n_files = 0
    expected = cfg.expected_size
    for rec in sorted(records, key=lambda r: r.image_id):
        n_files += 1
        rel_path = cfg.relative_to_root(rec.path)
        try:
            with Image.open(rec.path) as img:
                width, height = img.size
                mode = img.mode
        except OSError as exc:
            failed.append(_failure("image", rel_path, rec, exc))
            continue

        raw_boxes = None
        if adapter.label_path(rec).exists():
            n_files += 1
            try:
                raw_boxes = adapter.labels(rec)
            except ValueError as exc:
                # The image stays in the table, but without readable labels it is "unlabeled".
                failed.append(
                    _failure("xml", cfg.relative_to_root(adapter.label_path(rec)), rec, exc)
                )

        if raw_boxes is None:
            status = "unlabeled"
        else:
            # A label file with objects is "labeled" even if clipping later drops every box.
            status = "labeled" if raw_boxes else "labeled_no_defects"
            for box in raw_boxes:
                mapping.lookup(box.source_class)  # unmapped class stops the ingest
            clipped = clip_boxes(raw_boxes, width, height)
            dropped.update(clipped.dropped_classes)
            for box in clipped.boxes:
                phase_category, fine_class = mapping.lookup(box.source_class)
                box_rows.append(
                    {
                        "box_id": f"{rec.image_id}#{box.index}",
                        "image_id": rec.image_id,
                        "xmin": box.xmin,
                        "ymin": box.ymin,
                        "xmax": box.xmax,
                        "ymax": box.ymax,
                        "source_class": box.source_class,
                        "phase_category": phase_category,
                        "fine_class": fine_class,
                        "clipped": box.clipped,
                    }
                )
        image_rows.append(
            {
                "image_id": rec.image_id,
                "dataset": cfg.dataset,
                "build_id": rec.build_id,
                "modality": rec.modality,
                "layer_index": rec.layer_index,
                "timestamp": rec.timestamp,
                "z_mm": rec.z_mm,
                "path": rel_path,
                "width": width,
                "height": height,
                "mode": mode,
                "size_flag": (width, height) != expected[rec.modality],
                "label_status": status,
                "use_for_detection": status != "unlabeled",
            }
        )
    box_rows.sort(key=lambda r: (r["image_id"], int(r["box_id"].rsplit("#", 1)[1])))
    return image_rows, box_rows, failed, dropped, n_files


def _failure(kind: str, path: str, rec: ImageRecord, exc: Exception) -> dict:
    return {
        "kind": kind,
        "path": path,
        "build_id": rec.build_id,
        "modality": rec.modality,
        "reason": type(exc).__name__,
    }


def _write_outputs(cfg, adapter, mapping, tmp, in_hash, image_rows, box_rows, failed, dropped):
    outputs = {
        "images": write_table(tmp / "images.parquet", image_rows, tables.IMAGES),
        "boxes": write_table(tmp / "boxes.parquet", box_rows, tables.BOXES),
        "class_mapping": write_table(
            tmp / "class_mapping.parquet",
            [
                {
                    "dataset": mapping.dataset,
                    "source_class": source_class,
                    "phase_category": phase,
                    "fine_class": fine,
                    "mapping_version": mapping.version,
                }
                for source_class, (phase, fine) in sorted(mapping.classes.items())
            ],
            tables.CLASS_MAPPING,
        ),
    }

    con = open_tables(tmp, ["images"])
    try:
        pairs = pair_layers(con)
    finally:
        con.close()
    outputs["layer_pairs"] = write_table(
        tmp / "layer_pairs.parquet", pairs.table, tables.LAYER_PAIRS
    )

    builds_present = sorted({r["build_id"] for r in image_rows})
    outputs["folds"] = write_table(tmp / "folds.parquet", make_folds(builds_present), tables.FOLDS)

    con = open_tables(tmp, list(TABLE_NAMES))
    try:
        report = report_mod.build_report(
            con,
            builds=cfg.selected_builds,
            modalities=cfg.selected_modalities,
            source_classes=sorted(mapping.classes),
            dropped_boxes=dropped,
            unparsed_files=[cfg.relative_to_root(p) for p in adapter.unparsed],
            unknown_build_files=[cfg.relative_to_root(p) for p in adapter.unknown_build],
            orphan_xml=[cfg.relative_to_root(p) for p in adapter.orphan_xml()],
            failed_files=failed,
            duplicates=pairs.duplicates,
            notes=fold_notes(builds_present),
        )
    finally:
        con.close()

    status_counts = Counter(r["label_status"] for r in image_rows)
    metrics = {
        "images": len(image_rows),
        "labeled": status_counts["labeled"],
        "labeled_no_defects": status_counts["labeled_no_defects"],
        "unlabeled": status_counts["unlabeled"],
        "boxes": len(box_rows),
        "paired_layers": pairs.paired_layers,
        "folds": len({r["outer_fold"] for r in make_folds(builds_present)}),
    }
    manifest = build_manifest(
        dataset=cfg.dataset,
        source=adapter.info(),
        adapter_version=adapter.version,
        input_hash_value=in_hash,
        config=cfg.raw,
        mapping_version=mapping.version,
        seed=cfg.seed,
        outputs=outputs,
        metrics=metrics,
    )
    write_json(tmp / "ingest_report.json", report)
    write_json(tmp / "manifest.json", manifest)
    return IngestOutcome("written", manifest, report)

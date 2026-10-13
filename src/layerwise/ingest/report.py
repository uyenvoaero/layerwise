from collections import Counter
from typing import Any

import duckdb

from layerwise.ingest.sqlfiles import load_sql

IMAGE_COUNT_KEYS = ("images", "labeled", "labeled_no_defects", "unlabeled", "size_flagged")


def build_report(
    con: duckdb.DuckDBPyConnection,
    *,
    builds: list[str],
    modalities: list[str],
    source_classes: list[str],
    dropped_boxes: Counter,
    unparsed_files: list[str],
    unknown_build_files: list[str],
    orphan_xml: list[str],
    failed_files: list[dict[str, Any]],
    duplicates: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build the ingest report from DuckDB aggregates over the written Parquet tables."""
    per_build: dict[str, dict[str, dict[str, int]]] = {
        build: {
            modality: dict.fromkeys((*IMAGE_COUNT_KEYS, "failed", "unpaired", "duplicated"), 0)
            for modality in modalities
        }
        for build in builds
    }
    for build, modality, *counts in con.execute(
        load_sql("report_counts.sql", "images_by_build")
    ).fetchall():
        per_build[build][modality].update(
            zip(IMAGE_COUNT_KEYS, (int(c) for c in counts), strict=True)
        )
    for failed in failed_files:
        if failed["build_id"] in per_build:
            per_build[failed["build_id"]][failed["modality"]]["failed"] += 1
    for build, modality, count in con.execute(
        load_sql("report_counts.sql", "unpaired_by_build")
    ).fetchall():
        if build in per_build and modality in per_build[build]:
            per_build[build][modality]["unpaired"] = int(count)
    for dup in duplicates or []:
        per_build[dup["build_id"]][dup["modality"]]["duplicated"] += 1

    boxes: dict[str, dict[str, int]] = {
        name: {"ingested": 0, "clipped": 0, "dropped": 0} for name in source_classes
    }
    for source_class, ingested, clipped in con.execute(
        load_sql("report_counts.sql", "boxes_by_class")
    ).fetchall():
        boxes.setdefault(source_class, {"ingested": 0, "clipped": 0, "dropped": 0}).update(
            ingested=int(ingested), clipped=int(clipped)
        )
    for source_class, count in dropped_boxes.items():
        boxes.setdefault(source_class, {"ingested": 0, "clipped": 0, "dropped": 0})["dropped"] = (
            count
        )

    boxes_by_phase = {
        phase: int(count)
        for phase, count in con.execute(load_sql("report_counts.sql", "boxes_by_phase")).fetchall()
    }

    fold_columns = ("level", "outer_fold", "inner_fold", "role", "builds", "images")
    fold_columns += ("labeled_images", "boxes")
    fold_counts = [
        dict(zip(fold_columns, row, strict=True))
        for row in con.execute(load_sql("fold_counts.sql")).fetchall()
    ]

    return {
        "per_build": per_build,
        "boxes": boxes,
        "boxes_by_phase_category": boxes_by_phase,
        "unparsed_files": sorted(unparsed_files),
        "unknown_build_files": sorted(unknown_build_files),
        "orphan_xml": sorted(orphan_xml),
        "duplicates": duplicates or [],
        "failed_files": [
            {"kind": f["kind"], "path": f["path"], "reason": f["reason"]}
            for f in sorted(failed_files, key=lambda f: (f["path"], f["kind"]))
        ],
        "fold_counts": fold_counts,
        "notes": notes or [],
    }


def format_summary(report: dict[str, Any], metrics: dict[str, int]) -> str:
    """Short text summary: images, labeled/unlabeled per modality, pairs, boxes, folds."""
    totals: dict[str, Counter] = {}
    for modalities in report["per_build"].values():
        for modality, counts in modalities.items():
            totals.setdefault(modality, Counter()).update(
                {k: counts[k] for k in ("images", "labeled", "labeled_no_defects", "unlabeled")}
            )
    lines = [f"images: {metrics['images']}"]
    for modality in sorted(totals):
        t = totals[modality]
        lines.append(
            f"  {modality}: {t['images']} images, {t['labeled'] + t['labeled_no_defects']} labeled "
            f"({t['labeled_no_defects']} without defects), {t['unlabeled']} unlabeled"
        )
    lines.append(f"boxes: {metrics['boxes']}")
    for name, counts in report["boxes"].items():
        lines.append(
            f"  {name}: {counts['ingested']} ingested, {counts['clipped']} clipped, "
            f"{counts['dropped']} dropped"
        )
    lines.append(f"paired layers: {metrics['paired_layers']}")
    lines.append(f"outer folds: {metrics['folds']}")
    if report["failed_files"]:
        lines.append(f"failed files: {len(report['failed_files'])}")
    return "\n".join(lines)

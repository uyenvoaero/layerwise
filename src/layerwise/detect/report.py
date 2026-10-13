"""Comparison of all finished models and the baseline: Parquet table, Markdown and report.json."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from layerwise.detect import metrics
from layerwise.detect.config import MODELS, DetectConfig
from layerwise.detect.data import (
    labeled_counts,
    labeled_images,
    load_fold_plan,
)
from layerwise.detect.jobs import LABEL_SOURCE, read_json
from layerwise.detect.protocol import load_detection_evals
from layerwise.errors import ConfigError
from layerwise.ingest.sqlfiles import load_sql, open_tables
from layerwise.manifest import write_json

BASELINE = "contrast"
SQL_DIR = Path(__file__).parent / "sql"
BUILD_METRICS = ("map50", "recall", "precision", "f1")
SIZE_METRICS = ("recall_small", "recall_medium", "recall_large")
ALL_METRICS = (*BUILD_METRICS, *SIZE_METRICS)
METRIC_LABELS = {"map50": "mAP@0.5", "recall": "recall", "precision": "precision", "f1": "F1"}

COMPARISON_SCHEMA = pa.schema(
    [
        pa.field("model", pa.string(), nullable=False),
        pa.field("scope", pa.string(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("labeled_images", pa.int32(), nullable=False),
        pa.field("labeled_boxes", pa.int32(), nullable=False),
        pa.field("threshold", pa.float64(), nullable=True),
        pa.field("label_source", pa.string(), nullable=False),
        pa.field("verdict_vs_baseline", pa.string(), nullable=False),
        pa.field("partial", pa.bool_(), nullable=False),
    ]
)


@dataclass
class ReportResult:
    rows: list[dict[str, Any]]
    report: dict[str, Any]
    markdown: str


def verdict(value: float | None, baseline: float | None, tolerance: float) -> str:
    """`better` / `worse` beyond the tolerance, `equal` within it; no verdict for null values."""
    if value is None or baseline is None:
        return "not_applicable"
    if value - baseline > tolerance + 1e-12:
        return "better"
    if baseline - value > tolerance + 1e-12:
        return "worse"
    return "equal"


def write_report(cfg: DetectConfig) -> ReportResult:
    result = build_report(cfg)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(result.rows, schema=COMPARISON_SCHEMA)
    pq.write_table(table, cfg.reports_dir / "comparison.parquet", compression="zstd")
    (cfg.reports_dir / "comparison.md").write_text(result.markdown, encoding="utf-8", newline="\n")
    write_json(cfg.reports_dir / "report.json", result.report)
    return result


def build_report(cfg: DetectConfig) -> ReportResult:
    plan = load_fold_plan(cfg)
    n_folds = len(plan.outer_folds)
    builds = [f.test_build for f in plan.outer_folds]
    finished = _finished_folds(cfg, n_folds)
    if not finished:
        raise ConfigError(f"no finished results under {cfg.runs_dir}; run `layerwise train` first")
    if BASELINE not in finished:
        raise ConfigError(
            f"the baseline results are missing: run `layerwise baseline` first "
            f"(no finished fold for model {BASELINE!r})"
        )

    rows = _build_rows(cfg, finished, builds, n_folds)
    counts = labeled_counts(cfg)
    report: dict[str, Any] = {
        "models": sorted(finished),
        "label_source": LABEL_SOURCE,
        "folds_finished": {m: sorted(finished[m]) for m in sorted(finished)},
        "partial": any(len(finished[m]) < n_folds for m in finished),
        "inner_validation_flags": _inner_flags(cfg, finished),
        "builds_without_labeled_boxes": [
            b for b in builds if counts.get(b, {"boxes": 0})["boxes"] == 0
        ],
        "size_flagged_images": _size_flags(cfg),
        "reproduction": _reproduction(cfg, finished),
    }
    report["default_model"], report["default_model_reason"] = _default_model(
        cfg, rows, finished, n_folds
    )
    report_counts = {b: counts.get(b, {"images": 0})["images"] for b in builds}
    markdown = _markdown(cfg, rows, report, n_folds, report_counts)
    return ReportResult(rows, report, markdown)


# -- inputs ------------------------------------------------------------------------------------


def _finished_folds(cfg: DetectConfig, n_folds: int) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for model in MODELS:
        for k in range(n_folds):
            path = cfg.runs_dir / model / f"outer_{k}" / "heldout_result.json"
            if path.exists():
                out.setdefault(model, {})[k] = read_json(path)
    return out


def _fold_result_table(finished: dict[str, dict[int, dict[str, Any]]]) -> pa.Table:
    columns: dict[str, list] = {
        k: []
        for k in (
            "model", "outer_fold", "build_id", "images", "boxes", "predictions", "matches",
            "boxes_small", "hits_small", "boxes_medium", "hits_medium", "boxes_large", "hits_large",
        )
    }  # fmt: skip
    for model, folds in finished.items():
        for k, result in sorted(folds.items()):
            counts = result["counts"]
            columns["model"].append(model)
            columns["outer_fold"].append(k)
            columns["build_id"].append(result["held_out_build"])
            for name in ("images", "boxes", "predictions", "matches"):
                columns[name].append(counts[name])
            for bucket in ("small", "medium", "large"):
                entry = result["recall_by_size"][bucket]
                columns[f"boxes_{bucket}"].append(entry["boxes"])
                recall = entry["recall"]
                columns[f"hits_{bucket}"].append(
                    0 if recall is None else int(round(recall * entry["boxes"]))
                )
    return pa.table(columns)


def _pooled_map50(cfg: DetectConfig, folds: dict[int, dict[str, Any]], model: str) -> float | None:
    matches = []
    for k, result in sorted(folds.items()):
        images = labeled_images(cfg, [result["held_out_build"]])
        path = cfg.runs_dir / model / f"outer_{k}" / "heldout_detections.parquet"
        matches.append(metrics.match_images(load_detection_evals(path, images), cfg.match_iou))
    return metrics.average_precision(metrics.concat_matches(matches))


# -- rows --------------------------------------------------------------------------------------


def _build_rows(cfg, finished, builds, n_folds) -> list[dict[str, Any]]:
    values: dict[tuple[str, str, str], dict[str, Any]] = {}

    def put(model, scope, metric, value, images, boxes, threshold, partial):
        values[(model, scope, metric)] = {
            "model": model,
            "scope": scope,
            "metric": metric,
            "value": value,
            "labeled_images": int(images),
            "labeled_boxes": int(boxes),
            "threshold": threshold,
            "label_source": LABEL_SOURCE,
            "verdict_vs_baseline": "not_applicable",
            "partial": partial,
        }

    for model, folds in finished.items():
        for result in folds.values():
            scope = f"build:{result['held_out_build']}"
            counts = result["counts"]
            for metric in BUILD_METRICS:
                put(model, scope, metric, result[metric], counts["images"], counts["boxes"],
                    None if metric == "map50" else result["threshold"], False)  # fmt: skip
            for bucket in ("small", "medium", "large"):
                put(model, scope, f"recall_{bucket}", result["recall_by_size"][bucket]["recall"],
                    counts["images"], counts["boxes"], result["threshold"], False)  # fmt: skip

    con = duckdb.connect()
    try:
        con.register("fold_results", _fold_result_table(finished))
        cursor = con.execute(load_sql("comparison.sql", "pooled", SQL_DIR))
        names = [d[0] for d in cursor.description]
        pooled = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        con.close()
    for row in pooled:
        model = row["model"]
        partial = row["folds"] < n_folds
        for metric in ("recall", "precision", "f1", *SIZE_METRICS):
            put(model, "pooled", metric, row[metric], row["images"], row["boxes"], None, partial)
        put(model, "pooled", "map50", _pooled_map50(cfg, finished[model], model),
            row["images"], row["boxes"], None, partial)  # fmt: skip

    for (model, scope, metric), row in values.items():
        if model == BASELINE:
            continue
        base = values.get((BASELINE, scope, metric))
        if base is None:
            continue
        if scope == "pooled" and set(finished[model]) != set(finished[BASELINE]):
            continue  # pooled over different folds is not comparable
        row["verdict_vs_baseline"] = verdict(row["value"], base["value"], cfg.repro_tolerance)
    return sorted(values.values(), key=lambda r: (r["model"], r["scope"], r["metric"]))


def _inner_flags(cfg, finished) -> dict[str, dict[str, str]]:
    flags: dict[str, dict[str, str]] = {}
    for model in sorted(finished):
        for k in sorted(finished[model]):
            if str(k) in flags:
                continue
            selection = cfg.runs_dir / model / f"outer_{k}" / "selection.json"
            flags[str(k)] = dict(read_json(selection)["inner_validation_flags"])
    return dict(sorted(flags.items()))


def _size_flags(cfg: DetectConfig) -> dict[str, int]:
    con = open_tables(cfg.ingest.processed_dir, ["images"])
    try:
        rows = con.execute(load_sql("comparison.sql", "size_flags", SQL_DIR)).fetchall()
    finally:
        con.close()
    return {build: int(flagged) for build, flagged in rows}


def _reproduction(cfg, finished) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    tolerance = cfg.repro_tolerance
    for model in sorted(finished):
        for k, original in sorted(finished[model].items()):
            path = cfg.runs_dir / model / f"outer_{k}" / "repro" / "heldout_result.json"
            if not path.exists():
                continue
            rerun = read_json(path)
            diffs = [
                abs(original[m] - rerun[m])
                for m in ("map50", "recall")
                if original[m] is not None and rerun[m] is not None
            ]
            out.setdefault(model, {})[str(k)] = {
                "map50_original": original["map50"],
                "map50_rerun": rerun["map50"],
                "recall_original": original["recall"],
                "recall_rerun": rerun["recall"],
                "max_abs_diff": max(diffs) if diffs else None,
                "tolerance": tolerance,
                "reproduced": all(
                    metrics.within_tolerance(original[m], rerun[m], tolerance)
                    for m in ("map50", "recall")
                ),
            }
    return out


def _default_model(cfg, rows, finished, n_folds) -> tuple[str | None, str | None]:
    """Best eligible model by pooled mAP@0.5; never the baseline; null with a reason otherwise."""
    if any(len(folds) < n_folds for folds in finished.values()):
        return None, "partial_results"
    latency_path = cfg.reports_dir / "latency.json"
    records = read_json(latency_path) if latency_path.exists() else []
    target = cfg.latency.get("target_hardware", "cpu")
    candidates = {m for m in finished if m != BASELINE}
    measured = {r["model"] for r in records if r["hardware"] == target} & candidates
    if not measured:
        return None, "latency_not_measured"
    eligible = {
        r["model"] for r in records if r["hardware"] == target and r.get("eligible_default")
    } & candidates
    pooled = {
        r["model"]: r["value"]
        for r in rows
        if r["scope"] == "pooled" and r["metric"] == "map50" and r["value"] is not None
    }
    eligible &= set(pooled)
    if not eligible:
        return None, "none_eligible"
    return max(sorted(eligible), key=lambda m: pooled[m]), None


# -- markdown ----------------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "not defined" if value is None else f"{value:.3f}"


def _markdown(cfg, rows, report, n_folds, images_per_build) -> str:
    tolerance = cfg.repro_tolerance
    lines = [
        "# Powder bed anomaly detection: comparison",
        "",
        "Label source: Aalto manual PB boxes (recoating-phase anomaly; not porosity). "
        f"Matching: one-to-one, IoU {cfg.match_iou}. A verdict is `equal` within {tolerance} "
        "absolute. Generated from `comparison.parquet`; do not edit.",
        "",
    ]
    finished = report["folds_finished"]
    if report["partial"]:
        lines += ["**Partial results.** Pooled figures cover the finished folds only.", ""]
    for model in report["models"]:
        missing = [str(k) for k in range(n_folds) if k not in finished[model]]
        text = f"- {model}: finished folds {', '.join(map(str, finished[model]))}"
        lines.append(text + (f"; missing folds: {', '.join(missing)}" if missing else ""))
    not_run = [m for m in MODELS if m not in report["models"]]
    if not_run:
        lines.append(f"- Models not run: {', '.join(not_run)}")
    lines.append("")

    by_key = {(r["model"], r["scope"], r["metric"]): r for r in rows}
    scopes = sorted({r["scope"] for r in rows}, key=lambda s: (s == "pooled", s))
    for scope in scopes:
        any_row = next(r for r in rows if r["scope"] == scope)
        partial = " (partial)" if any_row["partial"] else ""
        lines += [f"## {scope}{partial}", ""]
        lines.append("| model | images | boxes | " + " | ".join(ALL_METRICS) + " |")
        lines.append("|---|---|---|" + "---|" * len(ALL_METRICS))
        for model in report["models"]:
            if (model, scope, "recall") not in by_key:
                continue
            first = by_key[(model, scope, "recall")]
            cells = []
            for metric in ALL_METRICS:
                row = by_key[(model, scope, metric)]
                tag = "" if model == BASELINE else f" ({row['verdict_vs_baseline']})"
                cells.append(_fmt(row["value"]) + tag)
            lines.append(
                f"| {model} | {first['labeled_images']} | {first['labeled_boxes']} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")

    lines += ["## Against the baseline", ""]
    for model in report["models"]:
        if model == BASELINE:
            continue
        for metric in BUILD_METRICS:
            row = by_key.get((model, "pooled", metric))
            if row is None:
                continue
            label = METRIC_LABELS[metric]
            note = " (partial)" if row["partial"] else ""
            if row["verdict_vs_baseline"] in ("equal", "worse"):
                lines.append(
                    f"- {model} does not beat the baseline on pooled {label}{note}: "
                    f"{row['verdict_vs_baseline']}."
                )
            elif row["verdict_vs_baseline"] == "better":
                lines.append(f"- {model} beats the baseline on pooled {label}{note}.")
            else:
                lines.append(f"- {model}: no verdict on pooled {label}{note}.")
    lines.append("")

    lines += ["## Flags", ""]
    minimum = cfg.min_validation_images
    if images_per_build and all(n < minimum for n in images_per_build.values()):
        lines.append(
            f"- Plumbing check only: every build has fewer than {minimum} labeled images "
            f"({', '.join(str(n) for n in images_per_build.values())}), so these figures carry "
            "no meaningful accuracy."
        )
    for fold, flags in report["inner_validation_flags"].items():
        for inner, flag in flags.items():
            if flag == "unreliable":
                lines.append(
                    f"- Outer fold {fold}, inner fold {inner}: unreliable validation build "
                    "(fewer labeled images than the configured minimum)."
                )
    for build in report["builds_without_labeled_boxes"]:
        lines.append(f"- Build {build} has no labeled boxes: recall and mAP are not defined.")
    for build, count in report["size_flagged_images"].items():
        if count:
            lines.append(f"- Build {build}: {count} PB image(s) with an unexpected size.")
    if report["default_model"]:
        lines += ["", f"Default model for later features: {report['default_model']}."]
    else:
        lines += ["", f"No default model: {report['default_model_reason']}."]
    return "\n".join(lines).rstrip() + "\n"

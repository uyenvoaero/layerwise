import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.conftest import BUILD_A, BUILD_B, BUILD_C, DETECT_CONFIG

from layerwise.detect import data, metrics, protocol, report
from layerwise.detect.config import load_detect_config
from layerwise.errors import ConfigError
from layerwise.ingest.pipeline import run_ingest
from layerwise.manifest import write_json

GT_BOX = [10, 10, 50, 40]
FALSE_ALARM = [200, 200, 240, 240]
BUILDS = sorted([BUILD_A, BUILD_B, BUILD_C])

SCHEMA = pa.schema(
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


@pytest.fixture
def cfg(synthetic, ingest_cfg):
    for build in BUILDS:
        for layer in range(1, 4):
            synthetic.add_pb(layer, boxes=[("defects", *GT_BOX)], build=build)
    run_ingest(ingest_cfg)
    return load_detect_config(DETECT_CONFIG, "fixture")


@pytest.fixture
def flagged_cfg(synthetic, ingest_cfg):
    for build in BUILDS:
        for layer in range(1, 4):
            synthetic.add_pb(layer, boxes=[("defects", *GT_BOX)], build=build)
    synthetic.add_pb(9, boxes=[("defects", *GT_BOX)], build=BUILDS[0], size=(100, 100))
    run_ingest(ingest_cfg)
    return load_detect_config(DETECT_CONFIG, "fixture")


def write_fold(cfg, model, k, *, hits, threshold, hit_score=0.8, fa_score=None, flags=("ok", "ok")):
    build = BUILDS[k]
    images = data.labeled_images(cfg, [build])
    evals, rows = [], []
    for index, image in enumerate(images):
        preds = []
        if index < hits:
            preds.append((GT_BOX, hit_score))
        if fa_score is not None:
            preds.append((FALSE_ALARM, fa_score))
        preds.sort(key=lambda p: -p[1])
        evals.append(
            metrics.ImageEval(
                image.image_id,
                image.boxes,
                [p[0] for p in preds],
                [p[1] for p in preds],
            )
        )
        rows += [(image.image_id, n, *map(float, box), s) for n, (box, s) in enumerate(preds)]
    match = metrics.match_images(evals, 0.5)
    result = protocol.build_heldout_result(
        match, threshold, build, f"{model}/c", 0.5, cfg.size_buckets
    )
    root = cfg.runs_dir / model / f"outer_{k}"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "heldout_result.json", result)
    write_json(
        root / "selection.json",
        {
            "winner": "c",
            "epochs": 1,
            "threshold": threshold,
            "inner_scores": {},
            "inner_std": 0.0,
            "inner_validation_flags": {"0": flags[0], "1": flags[1]},
        },
    )
    protocol.write_heldout_detections(root / "heldout_detections.parquet", rows, k)
    return result


def write_models(cfg, folds=(0, 1, 2)):
    for k in folds:
        thresholds = {0: 0.5, 1: 0.3, 2: 0.5}
        write_fold(cfg, "yolo", k, hits=3, threshold=thresholds[k], fa_score=0.4)
        write_fold(cfg, "contrast", k, hits=1, threshold=0.5, hit_score=0.7)


def rows_by_key(rows):
    return {(r["model"], r["scope"], r["metric"]): r for r in rows}


def test_comparison_columns_types_and_order_follow_the_contract(cfg):
    write_models(cfg)
    report.write_report(cfg)
    table = pq.read_table(cfg.reports_dir / "comparison.parquet")
    assert table.schema.equals(SCHEMA)
    rows = table.to_pylist()
    assert rows == sorted(rows, key=lambda r: (r["model"], r["scope"], r["metric"]))
    assert {r["metric"] for r in rows} == {
        "map50", "recall", "precision", "f1", "recall_small", "recall_medium", "recall_large",
    }  # fmt: skip
    assert {r["label_source"] for r in rows} == {"aalto_manual_pb_boxes"}


def test_pooled_scope_uses_the_union_with_each_folds_own_threshold(cfg):
    write_models(cfg)
    result = report.build_report(cfg)
    rows = rows_by_key(result.rows)
    # yolo: 9 hits, 3 false alarms kept at fold 1's lower threshold (12 predictions)
    assert rows[("yolo", "pooled", "precision")]["value"] == pytest.approx(9 / 12)
    assert rows[("yolo", "pooled", "recall")]["value"] == pytest.approx(1.0)
    assert rows[("yolo", "pooled", "f1")]["value"] == pytest.approx(2 * 0.75 / 1.75)
    assert rows[("yolo", "pooled", "map50")]["value"] == pytest.approx(1.0)
    assert rows[("yolo", "pooled", "map50")]["threshold"] is None
    assert rows[("yolo", "pooled", "recall")]["labeled_images"] == 9
    assert rows[("yolo", "pooled", "recall")]["labeled_boxes"] == 9
    assert rows[("contrast", "pooled", "recall")]["value"] == pytest.approx(1 / 3)
    assert rows[("contrast", "pooled", "map50")]["value"] == pytest.approx(1 / 3)
    assert rows[("yolo", f"build:{BUILDS[1]}", "precision")]["threshold"] == 0.3


def test_pooled_sql_aggregate_agrees_with_numpy_pooling(cfg):
    write_models(cfg)
    images = {b: data.labeled_images(cfg, [b]) for b in BUILDS}
    matches = []
    for k, build in enumerate(BUILDS):
        path = cfg.runs_dir / "yolo" / f"outer_{k}" / "heldout_detections.parquet"
        matches.append(
            metrics.match_images(protocol.load_detection_evals(path, images[build]), 0.5)
        )
    pooled = metrics.pooled_operating_point(matches, [0.5, 0.3, 0.5])
    rows = rows_by_key(report.build_report(cfg).rows)
    assert rows[("yolo", "pooled", "precision")]["value"] == pytest.approx(pooled["precision"])
    assert rows[("yolo", "pooled", "recall")]["value"] == pytest.approx(pooled["recall"])


def test_verdicts_use_the_reproduction_tolerance(cfg):
    assert report.verdict(0.60, 0.50, 0.02) == "better"
    assert report.verdict(0.40, 0.50, 0.02) == "worse"
    assert report.verdict(0.51, 0.50, 0.02) == "equal"
    assert report.verdict(0.52, 0.50, 0.02) == "equal"  # exactly at the tolerance
    assert report.verdict(None, 0.50, 0.02) == "not_applicable"
    assert report.verdict(0.50, None, 0.02) == "not_applicable"


def test_verdicts_per_metric_against_the_baseline(cfg):
    write_models(cfg)
    rows = rows_by_key(report.build_report(cfg).rows)
    assert rows[("yolo", "pooled", "recall")]["verdict_vs_baseline"] == "better"
    assert rows[("yolo", "pooled", "map50")]["verdict_vs_baseline"] == "better"
    assert rows[("yolo", "pooled", "precision")]["verdict_vs_baseline"] == "worse"  # .75 < 1
    assert rows[("yolo", f"build:{BUILDS[0]}", "precision")]["verdict_vs_baseline"] == "equal"
    assert rows[("yolo", f"build:{BUILDS[1]}", "precision")]["verdict_vs_baseline"] == "worse"
    baseline = [r for r in report.build_report(cfg).rows if r["model"] == "contrast"]
    assert {r["verdict_vs_baseline"] for r in baseline} == {"not_applicable"}


def test_markdown_states_when_a_model_does_not_beat_the_baseline(cfg):
    write_models(cfg)
    markdown = report.build_report(cfg).markdown
    assert "does not beat the baseline" in markdown
    assert "pooled precision" in markdown
    assert "Aalto manual PB boxes" in markdown
    assert "recoating-phase anomaly" in markdown


def test_fewer_than_three_folds_gives_partial_pooled_rows(cfg):
    write_models(cfg, folds=(0,))
    result = report.build_report(cfg)
    pooled = [r for r in result.rows if r["scope"] == "pooled"]
    assert pooled and all(r["partial"] for r in pooled)
    assert not any(r["partial"] for r in result.rows if r["scope"] != "pooled")
    assert result.report["partial"] is True
    assert result.report["folds_finished"] == {"contrast": [0], "yolo": [0]}
    assert "partial" in result.markdown.lower()
    assert "missing folds: 1, 2" in result.markdown
    assert "frcnn" in result.markdown  # missing model listed
    assert result.report["default_model"] is None
    assert result.report["default_model_reason"] == "partial_results"
    rows = rows_by_key(result.rows)
    assert rows[("yolo", "pooled", "recall")]["labeled_images"] == 3


def test_report_json_keys_flags_and_counts(flagged_cfg):
    cfg = flagged_cfg
    write_fold(cfg, "yolo", 0, hits=3, threshold=0.5, flags=("unreliable", "ok"))
    write_fold(cfg, "contrast", 0, hits=1, threshold=0.5, flags=("unreliable", "ok"))
    out = report.build_report(cfg).report
    assert out["label_source"] == "aalto_manual_pb_boxes"
    assert out["models"] == ["contrast", "yolo"]
    assert out["inner_validation_flags"] == {"0": {"0": "unreliable", "1": "ok"}}
    assert out["builds_without_labeled_boxes"] == []
    assert out["size_flagged_images"] == {BUILDS[0]: 1, BUILDS[1]: 0, BUILDS[2]: 0}
    assert "reproduction" in out and out["reproduction"] == {}


def test_default_model_needs_latency_and_never_picks_the_baseline(cfg):
    write_models(cfg)
    assert report.build_report(cfg).report["default_model"] is None
    assert report.build_report(cfg).report["default_model_reason"] == "latency_not_measured"

    cfg.reports_dir.mkdir(parents=True)

    def record(model, eligible):
        return {"model": model, "hardware": "cpu", "eligible_default": eligible}

    latency = cfg.reports_dir / "latency.json"
    write_json(latency, [record("contrast", True), record("yolo", False)])
    out = report.build_report(cfg).report
    assert out["default_model"] is None and out["default_model_reason"] == "none_eligible"

    write_json(latency, [record("contrast", True), record("yolo", True)])
    out = report.build_report(cfg).report
    assert out["default_model"] == "yolo"


def test_reproduction_section_compares_the_rerun_with_the_original(cfg):
    write_models(cfg)
    original = json.loads((cfg.runs_dir / "yolo/outer_0/heldout_result.json").read_text("utf-8"))
    repro = cfg.runs_dir / "yolo/outer_0/repro"
    repro.mkdir()
    write_json(repro / "heldout_result.json", {**original, "map50": original["map50"] - 0.01})
    entry = report.build_report(cfg).report["reproduction"]["yolo"]["0"]
    assert entry["reproduced"] is True and entry["tolerance"] == 0.02
    assert entry["map50_original"] == original["map50"]
    assert entry["max_abs_diff"] == pytest.approx(0.01)

    write_json(repro / "heldout_result.json", {**original, "recall": original["recall"] - 0.03})
    entry = report.build_report(cfg).report["reproduction"]["yolo"]["0"]
    assert entry["reproduced"] is False and entry["max_abs_diff"] == pytest.approx(0.03)


def test_no_results_or_missing_baseline_is_a_config_error(cfg):
    with pytest.raises(ConfigError, match="no finished"):
        report.build_report(cfg)
    write_fold(cfg, "yolo", 0, hits=3, threshold=0.5)
    with pytest.raises(ConfigError, match="contrast"):
        report.build_report(cfg)

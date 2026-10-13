import copy
import dataclasses
import json

import numpy as np
import pytest
from tests.conftest import BUILD_A, BUILD_B, BUILD_C, DETECT_CONFIG

from layerwise.detect import protocol
from layerwise.detect.config import load_detect_config
from layerwise.detect.models.base import FitResult
from layerwise.errors import FoldOverlapError
from layerwise.ingest.pipeline import run_ingest

GT_BOX = [10, 10, 50, 40]  # the box every synthetic labeled image carries
BEST_EPOCH = {("good", 0): 3, ("good", 1): 5, ("bad", 0): 1, ("bad", 1): 1}


class FakeDetector:
    """Deterministic scores. `good` hits the labeled box (0.9) and adds a weak false alarm (0.3);
    `bad` only returns a box far from every label."""

    name = "fake"
    trainable = True
    calls: list = []

    def __init__(self, candidate, record):
        self.candidate = candidate
        self.record = record

    def fit(self, job, train_tiles, validation_tiles, resume):
        builds_train = {t.image_id.split("_")[0] for t in train_tiles}
        builds_val = {t.image_id.split("_")[0] for t in validation_tiles or []}
        final = job.directory.name == "final"
        inner = 0 if final else int(job.directory.parent.name.split("_")[-1])
        self.record.append(("fit", self.candidate.id, job.epochs, builds_train, builds_val))
        weights = job.directory / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        (weights / "best.pt").write_text("best")
        (weights / "last.pt").write_text("last")
        best = BEST_EPOCH.get((self.candidate.id, inner), job.epochs)
        return FitResult(best, job.epochs, weights / "best.pt", weights / "last.pt", [])

    def load(self, path):
        self.record.append(("load", self.candidate.id))

    def predict_tiles(self, tiles):
        if self.candidate.id == "good":
            boxes = np.array([GT_BOX, [200, 200, 240, 240]], float)
            scores = np.array([0.9, 0.3])
        else:
            boxes = np.array([[200, 200, 240, 240]], float)
            scores = np.array([0.8])
        return [(boxes, scores) for _ in tiles]

    def info(self):
        return {"model": "fake", "runtime": "fake", "precision": "fp32", "device": "cpu"}


@pytest.fixture
def cfg(synthetic, ingest_cfg):
    for build in (BUILD_A, BUILD_B, BUILD_C):
        for layer in range(1, 4):
            synthetic.add_pb(layer, boxes=[("defects", *GT_BOX)], build=build)
    run_ingest(ingest_cfg)
    base = load_detect_config(DETECT_CONFIG, "fixture")
    raw = copy.deepcopy(base.raw)
    raw["tiling"].update(tile_size=320, overlap=64)  # one tile per 320x256 image
    raw["models"]["yolo"]["candidates"] = [
        {"id": "bad", "tile_size": 320},
        {"id": "good", "tile_size": 320},
    ]
    raw["min_validation_images"] = 3
    return dataclasses.replace(base, raw=raw)


def run(cfg, record, fold=0, **kwargs):
    factory = lambda model, candidate: FakeDetector(candidate, record)  # noqa: E731
    return protocol.run_protocol(cfg, "yolo", factory, folds=[fold], **kwargs)


def test_winner_has_highest_mean_inner_map_and_epochs_is_rounded_mean_of_best_epochs(cfg):
    record: list = []
    summary = run(cfg, record)
    outcome = summary.outcomes[0]
    selection = json.loads((outcome.run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["winner"] == "good"
    assert selection["epochs"] == 4  # round(mean(3, 5))
    assert selection["inner_scores"]["good"]["0"]["map50"] == pytest.approx(1.0)
    assert selection["inner_scores"]["bad"]["0"]["map50"] == pytest.approx(0.0)


def test_threshold_maximises_f1_on_pooled_inner_validation_predictions(cfg):
    summary = run(cfg, [])
    selection = summary.outcomes[0].selection
    # hits score 0.9, false alarms 0.3: F1 is highest when only the hits are kept
    assert selection["threshold"] == pytest.approx(0.9)


def test_held_out_build_is_never_seen_before_the_final_evaluation(cfg):
    record: list = []
    summary = run(cfg, record)
    outcome = summary.outcomes[0]
    held_out = outcome.heldout_build
    assert held_out == BUILD_A
    for entry in record:
        if entry[0] == "fit":
            assert held_out not in entry[3] and held_out not in entry[4]
    stages = [e["stage"] for e in outcome.events]
    assert stages.index("selection") < stages.index("final") < stages.index("heldout")
    assert stages.count("heldout") == 1


def test_heldout_result_has_the_contract_keys(cfg):
    summary = run(cfg, [])
    result = json.loads((summary.outcomes[0].run_dir / "heldout_result.json").read_text("utf-8"))
    assert result["held_out_build"] == BUILD_A
    assert result["label_source"] == "aalto_manual_pb_boxes"
    assert result["matching"] == {"one_to_one": True, "iou": 0.5}
    assert result["counts"]["images"] == 3 and result["counts"]["boxes"] == 3
    assert result["recall"] == pytest.approx(1.0) and result["map50"] == pytest.approx(1.0)
    assert (summary.outcomes[0].run_dir / "heldout_detections.parquet").exists()


def test_inner_validation_build_with_too_few_images_is_flagged_unreliable(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["min_validation_images"] = 4  # every build has 3 labeled images
    selection = run(dataclasses.replace(cfg, raw=raw), []).outcomes[0].selection
    assert selection["inner_validation_flags"] == {"0": "unreliable", "1": "unreliable"}
    assert selection["inner_std"] == pytest.approx(0.0)


def test_inner_validation_build_with_enough_images_is_ok(cfg):
    selection = run(cfg, []).outcomes[0].selection
    assert selection["inner_validation_flags"] == {"0": "ok", "1": "ok"}


def test_finished_jobs_are_skipped_on_rerun(cfg):
    first: list = []
    run(cfg, first)
    again: list = []
    summary = run(cfg, again)
    assert not [e for e in again if e[0] == "fit"]
    assert {e["status"] for e in summary.outcomes[0].events} == {"skipped"}


def test_interrupted_protocol_resumes_without_repeating_finished_jobs(cfg):
    record: list = []

    class Interrupt(Exception):
        pass

    def factory(model, candidate):
        detector = FakeDetector(candidate, record)
        original = detector.fit

        def fit(job, train_tiles, validation_tiles, resume):
            if len([r for r in record if r[0] == "fit"]) == 1:
                raise Interrupt
            return original(job, train_tiles, validation_tiles, resume)

        detector.fit = fit
        return detector

    with pytest.raises(Interrupt):
        protocol.run_protocol(cfg, "yolo", factory, folds=[0])
    fits_before = [r for r in record if r[0] == "fit"]
    assert len(fits_before) == 1
    summary = run(cfg, record)
    skipped = [e for e in summary.outcomes[0].events if e["status"] == "skipped"]
    assert len(skipped) == 1 and skipped[0]["stage"] == "inner"
    assert summary.outcomes[0].heldout_result is not None


def test_fold_table_with_a_build_in_two_roles_stops_before_any_job(cfg):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = cfg.ingest.processed_dir / "folds.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.append({**rows[0], "role": "test"})
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    record: list = []
    with pytest.raises(FoldOverlapError, match="roles"):
        run(cfg, record)
    assert record == []
    assert not cfg.runs_dir.exists()


def test_one_or_two_builds_cannot_run_the_full_protocol(synthetic, ingest_cfg):
    for build in (BUILD_A, BUILD_B):
        for layer in range(1, 4):
            synthetic.add_pb(layer, boxes=[("defects", *GT_BOX)], build=build)
    run_ingest(ingest_cfg)
    cfg = load_detect_config(DETECT_CONFIG, "fixture")
    record: list = []
    summary = protocol.run_protocol(
        cfg, "yolo", lambda m, c: FakeDetector(c, record), folds=None
    )
    assert summary.full_protocol is False
    assert "full protocol" in " ".join(summary.notes)
    assert summary.outcomes == [] and record == []


def test_dry_run_lists_jobs_and_writes_nothing(cfg):
    jobs = protocol.plan_jobs(cfg, "yolo", folds=[0])
    labels = [j["label"] for j in jobs]
    assert "yolo/outer_0/inner/inner_0/bad" in labels
    assert any(j["kind"] == "final" for j in jobs) and any(j["kind"] == "heldout" for j in jobs)
    assert not any(j["done"] for j in jobs)
    assert not cfg.runs_dir.exists()
    run(cfg, [])
    assert all(j["done"] for j in protocol.plan_jobs(cfg, "yolo", folds=[0]))

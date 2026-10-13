"""Nested leave-one-build-out protocol: inner search, threshold, final retrain, held-out evaluation.

Every model (baseline included) runs through `run_protocol` with the same matching and metrics.
Models without training run only the threshold step and the held-out evaluation.
"""

import dataclasses
import shutil
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from layerwise.detect import metrics, staging
from layerwise.detect.config import Candidate, DetectConfig, TilingParams
from layerwise.detect.data import (
    FoldPlan,
    LabeledImage,
    OuterFold,
    filter_readable,
    labeled_counts,
    labeled_images,
    load_fold_plan,
    read_gray,
)
from layerwise.detect.jobs import (
    LABEL_SOURCE,
    JobKey,
    finish_job,
    is_done,
    job_dir,
    now,
    read_json,
    run_root,
    staging_dir,
)
from layerwise.detect.models.base import Detector, FitJob, predict_image
from layerwise.errors import ConfigError
from layerwise.manifest import write_json
from layerwise.tracking import start_run

EXPERIMENT = "detect"
DetectorFactory = Callable[[str, Candidate], Detector]
Report = Callable[[str, str, str], None]  # stage, label, status


@dataclass
class FoldOutcome:
    outer_fold: int
    run_dir: Path
    heldout_build: str
    selection: dict[str, Any] | None = None
    heldout_result: dict[str, Any] | None = None
    events: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ProtocolSummary:
    outcomes: list[FoldOutcome]
    full_protocol: bool
    notes: list[str]


def _select_folds(plan: FoldPlan, folds: list[int] | None) -> list[OuterFold]:
    if folds is None:
        return plan.outer_folds
    known = {f.outer_fold: f for f in plan.outer_folds}
    missing = [k for k in folds if k not in known]
    if missing:
        raise ConfigError(f"outer fold(s) {missing} not in the fold table (have {sorted(known)})")
    return [known[k] for k in folds]


def plan_jobs(
    cfg: DetectConfig, model: str, folds: list[int] | None = None, repro: bool = False
) -> list[dict[str, Any]]:
    """The jobs a run would execute and whether each is already done; writes nothing."""
    plan = load_fold_plan(cfg)
    out = []
    for fold in _select_folds(plan, folds):
        root = run_root(cfg, model, fold.outer_fold, repro)
        for candidate in cfg.candidates(model):
            for inner in fold.inner:
                key = JobKey(model, fold.outer_fold, "inner", inner.inner_fold, candidate.id)
                out.append(_plan_row(key, is_done(job_dir(cfg, key, repro))))
        selection_path = root / "selection.json"
        winner = read_json(selection_path)["winner"] if selection_path.exists() else "<winner>"
        key = JobKey(model, fold.outer_fold, "final", None, winner)
        out.append(_plan_row(key, is_done(root / "final")))
        key = JobKey(model, fold.outer_fold, "heldout", None, winner)
        out.append(_plan_row(key, is_done(root, "heldout_")))
    return out


def _plan_row(key: JobKey, done: bool) -> dict[str, Any]:
    return {**key.as_dict(), "label": key.label(), "done": done}


def run_protocol(
    cfg: DetectConfig,
    model: str,
    factory: DetectorFactory,
    folds: list[int] | None = None,
    repro: bool = False,
    report: Report | None = None,
) -> ProtocolSummary:
    """Run the selected outer folds. The fold-overlap check runs first and stops on a violation."""
    plan = load_fold_plan(cfg)
    if not plan.full_protocol:
        n = len(plan.outer_folds)
        return ProtocolSummary(
            [],
            False,
            [
                f"full protocol was not possible: {n} build(s) with labeled folds present, "
                "3 builds with 2 inner folds each are needed; no results were produced"
            ],
        )
    selected = _select_folds(plan, folds)
    builds = sorted({b for f in plan.outer_folds for b in [f.test_build, *f.train_builds]})
    images = filter_readable(cfg, labeled_images(cfg, builds))
    by_build: dict[str, list[LabeledImage]] = defaultdict(list)
    for image in images:
        by_build[image.build_id].append(image)
    counts = labeled_counts(cfg)
    outcomes = [
        _run_fold(cfg, model, fold, factory, by_build, counts, repro, report or _silent)
        for fold in selected
    ]
    return ProtocolSummary(outcomes, True, [])


def _silent(stage: str, label: str, status: str) -> None:
    pass


def _run_fold(
    cfg: DetectConfig,
    model: str,
    fold: OuterFold,
    factory: DetectorFactory,
    by_build: dict[str, list[LabeledImage]],
    counts: dict[str, dict[str, int]],
    repro: bool,
    report: Report,
) -> FoldOutcome:
    root = run_root(cfg, model, fold.outer_fold, repro)
    outcome = FoldOutcome(fold.outer_fold, root, fold.test_build)

    def note(stage: str, label: str, status: str) -> None:
        outcome.events.append({"stage": stage, "label": label, "status": status})
        report(stage, label, status)

    ctx = _FoldContext(cfg, model, fold, factory, by_build, counts, repro, note)
    run_name = f"{model}/outer_{fold.outer_fold}" + (" repro" if repro else "")
    with start_run(cfg, EXPERIMENT, run_name) as parent:
        parent.log({"model": model, "held_out_build": fold.test_build, "repro": repro})
        if repro:
            outcome.selection = ctx.reuse_selection()
        else:
            outcome.selection = ctx.inner_search()
        winner = ctx.candidate(outcome.selection["winner"])
        ctx.final_job(winner, outcome.selection)
        outcome.heldout_result = ctx.heldout(winner, outcome.selection)
        parent.log(
            metrics={
                "heldout_map50": outcome.heldout_result["map50"],
                "heldout_recall": outcome.heldout_result["recall"],
                "heldout_precision": outcome.heldout_result["precision"],
            }
        )
    return outcome


class _FoldContext:
    """Jobs of one outer fold. The held-out build is only read in `heldout`."""

    def __init__(self, cfg, model, fold, factory, by_build, counts, repro, note):
        self.cfg = cfg
        self.model = model
        self.fold = fold
        self.factory = factory
        self.by_build = by_build
        self.counts = counts
        self.repro = repro
        self.note = note
        self.root = run_root(cfg, model, fold.outer_fold, repro)

    # -- helpers -----------------------------------------------------------------------------

    def candidate(self, candidate_id: str) -> Candidate:
        return next(c for c in self.cfg.candidates(self.model) if c.id == candidate_id)

    def images(self, builds: list[str]) -> list[LabeledImage]:
        return [image for build in sorted(builds) for image in self.by_build.get(build, [])]

    def resolved(self, candidate: Candidate, tp: TilingParams, **extra) -> dict[str, Any]:
        return {
            "config": self.cfg.raw,
            "profile": self.cfg.profile,
            "candidate": dataclasses.asdict(candidate),
            "tiling": dataclasses.asdict(tp),
            **extra,
        }

    def fit_job(self, key: JobKey, candidate: Candidate, tp: TilingParams, epochs: int) -> FitJob:
        train = self.cfg.train
        return FitJob(
            directory=job_dir(self.cfg, key, self.repro),
            staging_dir=staging_dir(self.cfg, key, self.repro),
            candidate=candidate,
            tiling=tp,
            epochs=epochs,
            patience=int(train["patience"]),
            seed=self.cfg.seed,
            train=train,
            options=self.cfg.model_options(self.model),
            weights_dir=self.cfg.weights_dir,
        )

    def predict(self, detector: Detector, images: list[LabeledImage], tp: TilingParams):
        """Image-level predictions: evaluation records and rows for the detections table."""
        floor = self.cfg.eval_min_confidence
        evals, rows = [], []
        for image in images:
            boxes, scores = predict_image(read_gray(image.path), detector, tp)
            keep = scores >= floor
            boxes, scores = boxes[keep], scores[keep]
            evals.append(metrics.ImageEval(image.image_id, image.boxes, boxes, scores))
            for n, (box, score) in enumerate(zip(boxes, scores, strict=True)):
                rows.append((image.image_id, n, *map(float, box), float(score)))
        return evals, rows

    # -- inner search ------------------------------------------------------------------------

    def inner_search(self) -> dict[str, Any]:
        candidates = self.cfg.candidates(self.model)
        results: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        all_skipped = True
        for candidate in candidates:
            for inner in self.fold.inner:
                key = JobKey(
                    self.model, self.fold.outer_fold, "inner", inner.inner_fold, candidate.id
                )
                directory = job_dir(self.cfg, key, self.repro)
                if is_done(directory):
                    results[candidate.id][inner.inner_fold] = read_json(directory / "result.json")
                    self.note("inner", key.label(), "skipped")
                    continue
                all_skipped = False
                results[candidate.id][inner.inner_fold] = self.inner_job(key, candidate, inner)
                self.note("inner", key.label(), "ran")

        selection_path = self.root / "selection.json"
        if all_skipped and selection_path.exists():
            self.note("selection", f"{self.model}/outer_{self.fold.outer_fold}", "skipped")
            return read_json(selection_path)
        selection = self.select(results)
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(selection_path, selection)
        self.note("selection", f"{self.model}/outer_{self.fold.outer_fold}", "ran")
        return selection

    def inner_job(self, key: JobKey, candidate: Candidate, inner) -> dict[str, Any]:
        directory = job_dir(self.cfg, key, self.repro)
        tp = self.cfg.tiling_for(candidate)
        train_images = self.images(inner.train_builds)
        validation_images = self.images(inner.validation_builds)
        detector = self.factory(self.model, candidate)
        started = now()
        with start_run(self.cfg, EXPERIMENT, key.label(), nested=True) as run:
            directory.mkdir(parents=True, exist_ok=True)
            best_epoch, epochs_run, weights = 0, 0, None
            if detector.trainable:
                fit = detector.fit(
                    self.fit_job(key, candidate, tp, int(self.cfg.train["max_epochs"])),
                    staging.build_tile_entries(train_images, tp, self.cfg.seed),
                    staging.build_tile_entries(validation_images, tp, self.cfg.seed),
                    resume=True,
                )
                best_epoch, epochs_run, weights = fit.best_epoch, fit.epochs_run, fit.best_weights
                detector.load(weights)
            else:
                detector.load(None)
            evals, rows = self.predict(detector, validation_images, tp)
            match = metrics.match_images(evals, self.cfg.match_iou)
            map50 = metrics.average_precision(match)
            _write_detections(directory / "val_detections.parquet", rows)
            result = {
                "candidate": candidate.id,
                "inner_fold": inner.inner_fold,
                "best_epoch": best_epoch,
                "epochs_run": epochs_run,
                "map50": map50,
                "labeled_images": sum(self.counts[b]["images"] for b in inner.validation_builds),
                "labeled_boxes": sum(self.counts[b]["boxes"] for b in inner.validation_builds),
                "train_builds": inner.train_builds,
                "validation_builds": inner.validation_builds,
                "best_weights": _relative(weights, directory),
            }
            if map50 is None:
                result["map50_reason"] = "not_defined"
            run.log(
                {
                    "candidate": candidate.id,
                    "tile_size": tp.tile_size,
                    "inner_fold": inner.inner_fold,
                },
                {"map50": map50, "best_epoch": best_epoch},
            )
            finish_job(
                self.cfg,
                key,
                directory,
                result=result,
                image_ids=[i.image_id for i in train_images + validation_images],
                train_builds=inner.train_builds,
                validation_builds=inner.validation_builds,
                test_builds=[],
                device=_device(detector),
                resolved_config=self.resolved(candidate, tp),
                tracking_run_id=run.run_id,
                started_at=started,
            )
        return result

    def select(self, results: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
        def mean_map(candidate_id: str) -> float:
            values = [r["map50"] for r in results[candidate_id].values() if r["map50"] is not None]
            return float(np.mean(values)) if values else -1.0

        candidates = self.cfg.candidates(self.model)
        winner = max(candidates, key=lambda c: (mean_map(c.id), -candidates.index(c)))
        won = results[winner.id]
        trainable = any(r["best_epoch"] > 0 for r in won.values())
        epochs = 0
        if trainable:
            mean_epoch = float(np.mean([r["best_epoch"] for r in won.values()]))
            epochs = max(1, int(np.floor(mean_epoch + 0.5)))

        matches = []
        for inner in self.fold.inner:
            key = JobKey(self.model, self.fold.outer_fold, "inner", inner.inner_fold, winner.id)
            evals = _load_evals(
                job_dir(self.cfg, key, self.repro) / "val_detections.parquet",
                self.images(inner.validation_builds),
            )
            matches.append(metrics.match_images(evals, self.cfg.match_iou))
        threshold = metrics.best_f1_threshold(metrics.concat_matches(matches))
        defined = [r["map50"] for r in won.values() if r["map50"] is not None]
        selection: dict[str, Any] = {
            "winner": winner.id,
            "epochs": epochs,
            "threshold": 0.5 if threshold is None else float(threshold),
            "inner_scores": {
                cid: {
                    str(j): {
                        k: r[k] for k in ("map50", "best_epoch", "labeled_images", "labeled_boxes")
                    }
                    for j, r in sorted(per_inner.items())
                }
                for cid, per_inner in results.items()
            },
            "inner_std": float(np.std(defined)) if defined else None,
            "inner_validation_flags": {
                str(inner.inner_fold): "unreliable"
                if sum(self.counts[b]["images"] for b in inner.validation_builds)
                < self.cfg.min_validation_images
                else "ok"
                for inner in self.fold.inner
            },
        }
        if threshold is None:
            selection["threshold_note"] = "no predictions or no labeled boxes; default 0.5 used"
        return selection

    def reuse_selection(self) -> dict[str, Any]:
        """`--repro`: rerun the fold from its finished selection, into a fresh `repro/` folder."""
        original = run_root(self.cfg, self.model, self.fold.outer_fold, False)
        if not (original / "selection.json").exists() or not is_done(original, "heldout_"):
            raise ConfigError(
                f"--repro needs a finished fold: {self.model}/outer_{self.fold.outer_fold} "
                "has no selection.json and heldout_result.json"
            )
        selection = read_json(original / "selection.json")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        write_json(self.root / "selection.json", selection)
        self.note("selection", f"{self.model}/outer_{self.fold.outer_fold}", "reused")
        return selection

    # -- final and held-out ------------------------------------------------------------------

    def final_job(self, candidate: Candidate, selection: dict[str, Any]) -> None:
        key = JobKey(self.model, self.fold.outer_fold, "final", None, candidate.id)
        directory = job_dir(self.cfg, key, self.repro)
        if is_done(directory):
            self.note("final", key.label(), "skipped")
            return
        tp = self.cfg.tiling_for(candidate)
        images = self.images(self.fold.train_builds)
        detector = self.factory(self.model, candidate)
        started = now()
        with start_run(self.cfg, EXPERIMENT, key.label(), nested=True) as run:
            directory.mkdir(parents=True, exist_ok=True)
            result: dict[str, Any] = {
                "candidate": candidate.id,
                "epochs": selection["epochs"],
                "trained": detector.trainable,
                "weights": None,
            }
            if detector.trainable:
                fit = detector.fit(
                    self.fit_job(key, candidate, tp, selection["epochs"]),
                    staging.build_tile_entries(images, tp, self.cfg.seed),
                    None,
                    resume=True,
                )
                result["epochs_run"] = fit.epochs_run
                result["weights"] = _relative(fit.last_weights, directory)
            run.log({"candidate": candidate.id, "epochs": selection["epochs"]})
            finish_job(
                self.cfg,
                key,
                directory,
                result=result,
                image_ids=[i.image_id for i in images],
                train_builds=self.fold.train_builds,
                validation_builds=[],
                test_builds=[],
                device=_device(detector),
                resolved_config=self.resolved(candidate, tp, epochs=selection["epochs"]),
                tracking_run_id=run.run_id,
                started_at=started,
            )
        self.note("final", key.label(), "ran")

    def heldout(self, candidate: Candidate, selection: dict[str, Any]) -> dict[str, Any]:
        key = JobKey(self.model, self.fold.outer_fold, "heldout", None, candidate.id)
        if is_done(self.root, "heldout_"):
            self.note("heldout", key.label(), "skipped")
            return read_json(self.root / "heldout_result.json")
        final_dir = self.root / "final"
        final = read_json(final_dir / "result.json")
        tp = self.cfg.tiling_for(candidate)
        images = self.images([self.fold.test_build])
        detector = self.factory(self.model, candidate)
        started = now()
        with start_run(self.cfg, EXPERIMENT, key.label(), nested=True) as run:
            detector.load(final_dir / final["weights"] if final["weights"] else None)
            evals, rows = self.predict(detector, images, tp)
            match = metrics.match_images(evals, self.cfg.match_iou)
            threshold = selection["threshold"]
            result = build_heldout_result(
                match,
                threshold,
                self.fold.test_build,
                f"{self.model}/{candidate.id}",
                self.cfg.match_iou,
                self.cfg.size_buckets,
            )
            write_heldout_detections(
                self.root / "heldout_detections.parquet", rows, self.fold.outer_fold
            )
            run.log(
                {"threshold": threshold, "held_out_build": self.fold.test_build},
                {k: result[k] for k in ("map50", "precision", "recall", "f1")},
            )
            finish_job(
                self.cfg,
                key,
                self.root,
                result=result,
                image_ids=[i.image_id for i in images],
                train_builds=self.fold.train_builds,
                validation_builds=[],
                test_builds=[self.fold.test_build],
                device=_device(detector),
                resolved_config=self.resolved(candidate, tp, threshold=threshold),
                tracking_run_id=run.run_id,
                started_at=started,
                metrics=result,
                prefix="heldout_",
            )
        self.note("heldout", key.label(), "ran")
        return result


def build_heldout_result(
    match: metrics.MatchResult,
    threshold: float,
    held_out_build: str,
    model_id: str,
    iou: float,
    buckets: dict[str, int],
) -> dict[str, Any]:
    """Contents of heldout_result.json (see contracts/run-outputs.md)."""
    point = metrics.operating_point(match, threshold)
    map50 = metrics.average_precision(match)
    result = {
        "held_out_build": held_out_build,
        "threshold": threshold,
        "map50": map50,
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
        "recall_by_size": metrics.recall_by_size(match, threshold, buckets),
        "counts": {
            k: point[k]
            for k in ("images", "boxes", "predictions", "matches", "false_alarms", "misses")
        },
        "matching": {"one_to_one": True, "iou": iou},
        "label_source": LABEL_SOURCE,
        "model_id": model_id,
    }
    if map50 is None:
        result["map50_reason"] = "not_defined"
    return result


def _relative(path: Path | None, directory: Path) -> str | None:
    return None if path is None else Path(path).relative_to(directory).as_posix()


def _device(detector: Detector) -> dict[str, Any]:
    info = detector.info()
    return {"type": info.get("device", "cpu"), "name": info.get("device_name")}


def _write_detections(path: Path, rows: list[tuple]) -> None:
    """Validation predictions of an inner job: image_id, box, confidence (sorted)."""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    table = pa.table(
        {
            "image_id": pa.array([r[0] for r in rows], pa.string()),
            "xmin": pa.array([r[2] for r in rows], pa.float64()),
            "ymin": pa.array([r[3] for r in rows], pa.float64()),
            "xmax": pa.array([r[4] for r in rows], pa.float64()),
            "ymax": pa.array([r[5] for r in rows], pa.float64()),
            "confidence": pa.array([r[6] for r in rows], pa.float64()),
        }
    )
    pq.write_table(table, path, compression="zstd")


def write_heldout_detections(path: Path, rows: list[tuple], outer_fold: int) -> None:
    items = [(f"{r[0]}#{r[1]}", r) for r in sorted(rows, key=lambda r: (r[0], r[1]))]
    table = pa.table(
        {
            "detection_id": pa.array([i for i, _ in items], pa.string()),
            "image_id": pa.array([r[0] for _, r in items], pa.string()),
            "xmin": pa.array([r[2] for _, r in items], pa.float64()),
            "ymin": pa.array([r[3] for _, r in items], pa.float64()),
            "xmax": pa.array([r[4] for _, r in items], pa.float64()),
            "ymax": pa.array([r[5] for _, r in items], pa.float64()),
            "confidence": pa.array([r[6] for _, r in items], pa.float64()),
            "outer_fold": pa.array([outer_fold] * len(items), pa.int32()),
        }
    )
    pq.write_table(table, path, compression="zstd")


def load_detection_evals(path: Path, images: list[LabeledImage]) -> list[metrics.ImageEval]:
    """Rebuild evaluation records from a stored detections table and the labeled images."""
    return _load_evals(path, images)


def _load_evals(path: Path, images: list[LabeledImage]) -> list[metrics.ImageEval]:
    table = pq.read_table(path).to_pydict()
    boxes: dict[str, list[list[float]]] = defaultdict(list)
    scores: dict[str, list[float]] = defaultdict(list)
    for i, image_id in enumerate(table["image_id"]):
        boxes[image_id].append(
            [table["xmin"][i], table["ymin"][i], table["xmax"][i], table["ymax"][i]]
        )
        scores[image_id].append(table["confidence"][i])
    return [
        metrics.ImageEval(
            image.image_id,
            image.boxes,
            np.array(boxes.get(image.image_id, []), float).reshape(-1, 4),
            np.array(scores.get(image.image_id, []), float),
        )
        for image in images
    ]

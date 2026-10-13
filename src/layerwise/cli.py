from enum import StrEnum
from typing import Annotated, NoReturn

import pyarrow.parquet as pq
import typer

from layerwise.config import DEFAULT_CONFIG, IngestConfig, load_config
from layerwise.detect.config import (
    DEFAULT_DETECT_CONFIG,
    DetectConfig,
    check_vision_extra,
    load_detect_config,
)
from layerwise.errors import (
    ConfigError,
    FoldOverlapError,
    InputErrorsExceededError,
    UnmappedClassError,
    UnreadableImagesError,
)
from layerwise.ingest.checks import find_violations
from layerwise.ingest.fixture import FixtureError, make_fixture
from layerwise.ingest.pipeline import run_ingest
from layerwise.ingest.report import format_summary
from layerwise.ingest.sqlfiles import load_sql, open_tables

EXIT_CHECK_FAILED = 1
EXIT_CONFIG = 2
EXIT_UNMAPPED = 3
EXIT_INPUT_ERRORS = 4
EXIT_OVER_BUDGET = 4  # detect commands: model over the latency budget

app = typer.Typer(help="Layerwise: L-PBF layer image ingest.", no_args_is_help=True)


class Profile(StrEnum):
    local = "local"
    aws = "aws"
    fixture = "fixture"


ConfigOption = Annotated[str | None, typer.Option("--config", help="Base config file.")]
ProfileOption = Annotated[Profile | None, typer.Option("--profile", help="Config profile.")]


@app.callback()
def main(ctx: typer.Context, config: ConfigOption = None, profile: ProfileOption = None) -> None:
    """Options given here apply to every command; a command may also repeat them."""
    ctx.obj = {"config": config, "profile": profile}


def load_ctx_config(
    ctx: typer.Context, config: str | None = None, profile: Profile | None = None
) -> IngestConfig:
    """Load the config chosen on the command line; a bad config exits with code 2."""
    config = config or ctx.obj["config"] or DEFAULT_CONFIG
    profile = profile or ctx.obj["profile"] or Profile.local
    try:
        return load_config(config, profile.value)
    except ConfigError as exc:
        fail(str(exc), EXIT_CONFIG)


def load_detect_ctx_config(
    ctx: typer.Context, config: str | None = None, profile: Profile | None = None
) -> DetectConfig:
    """Load the detect config chosen on the command line; a bad config or a missing `vision`
    extra exits with code 2."""
    config = config or ctx.obj["config"] or DEFAULT_DETECT_CONFIG
    profile = profile or ctx.obj["profile"] or Profile.local
    try:
        check_vision_extra()
        return load_detect_config(config, profile.value)
    except ConfigError as exc:
        fail(str(exc), EXIT_CONFIG)


def fail(message: str, code: int) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


EXPECTED_ERRORS = {
    ConfigError: EXIT_CONFIG,
    UnmappedClassError: EXIT_UNMAPPED,
    InputErrorsExceededError: EXIT_INPUT_ERRORS,
}


# Detect commands: 1 fold overlap, 2 configuration, 3 unreadable labeled images.
DETECT_ERRORS = {
    FoldOverlapError: EXIT_CHECK_FAILED,
    ConfigError: EXIT_CONFIG,
    UnreadableImagesError: EXIT_UNMAPPED,
}


def detect_exit_code_for(exc: Exception) -> int | None:
    for error_type, code in DETECT_ERRORS.items():
        if isinstance(exc, error_type):
            return code
    return None


def exit_code_for(exc: Exception) -> int | None:
    for error_type, code in EXPECTED_ERRORS.items():
        if isinstance(exc, error_type):
            return code
    return None


@app.command()
def ingest(
    ctx: typer.Context,
    config: ConfigOption = None,
    profile: ProfileOption = None,
    force: Annotated[bool, typer.Option("--force", help="Rewrite even if unchanged.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Scan and print; write nothing.")
    ] = False,
) -> None:
    """Scan the dataset and write the Parquet tables, report and manifest."""
    cfg = load_ctx_config(ctx, config, profile)
    try:
        outcome = run_ingest(cfg, force=force, dry_run=dry_run)
    except tuple(EXPECTED_ERRORS) as exc:
        fail(str(exc), exit_code_for(exc))
    if outcome.status == "up_to_date":
        typer.echo("up to date")
        return
    typer.echo(format_summary(outcome.report, outcome.manifest["metrics"]))


@app.command("make-fixture")
def make_fixture_command(
    ctx: typer.Context, config: ConfigOption = None, profile: ProfileOption = None
) -> None:
    """Build the committed smoke-test fixture from the full dataset."""
    cfg = load_ctx_config(ctx, config, profile)
    try:
        result = make_fixture(cfg)
    except ConfigError as exc:
        fail(str(exc), EXIT_CONFIG)
    except FixtureError as exc:
        fail(str(exc), EXIT_CHECK_FAILED)
    typer.echo(f"fixture written: {result.images} images, {result.total_bytes} bytes")


@app.command("check-splits")
def check_splits(
    ctx: typer.Context, config: ConfigOption = None, profile: ProfileOption = None
) -> None:
    """Validate folds.parquet: no build may have two roles in a fold."""
    cfg = load_ctx_config(ctx, config, profile)
    folds_path = cfg.processed_dir / "folds.parquet"
    if not folds_path.exists():
        fail(f"{folds_path} not found; run `layerwise ingest` first", EXIT_CONFIG)
    rows = pq.read_table(folds_path).to_pylist()
    violations = find_violations(rows)
    if violations:
        for violation in violations:
            typer.echo(violation.message, err=True)
        fail(f"{len(violations)} leakage violation(s) in {folds_path.name}", EXIT_CHECK_FAILED)

    outer = {r["outer_fold"] for r in rows if r["level"] == "outer"}
    inner_per_outer = {
        k: len({r["inner_fold"] for r in rows if r["level"] == "inner" and r["outer_fold"] == k})
        for k in sorted(outer)
    }
    inner_counts = sorted(set(inner_per_outer.values()))
    inner_text = (
        f"{inner_counts[0]} inner folds each"
        if len(inner_counts) == 1
        else "inner folds per outer fold: "
        + ", ".join(f"{k}: {n}" for k, n in inner_per_outer.items())
    )
    typer.echo(f"{len(outer)} outer folds, {inner_text}")
    typer.echo(f"valid: no build has more than one role in any fold ({len(rows)} rows)")
    con = open_tables(cfg.processed_dir, ["images", "boxes", "folds"])
    try:
        counts = con.execute(load_sql("fold_counts.sql")).fetchall()
    finally:
        con.close()
    for level, outer_fold, inner_fold, role, builds, images, labeled, boxes in counts:
        fold = f"{level} {outer_fold}" + ("" if inner_fold is None else f".{inner_fold}")
        typer.echo(
            f"  {fold:<12} {role:<11} builds={builds} images={images} "
            f"labeled_images={labeled} boxes={boxes}"
        )


ModelOption = Annotated[str, typer.Option("--model", help="yolo (default), frcnn or contrast.")]
FoldOption = Annotated[int | None, typer.Option("--fold", help="Run only this outer fold.")]


def _fmt(value: float | None) -> str:
    return "not defined" if value is None else f"{value:.3f}"


def format_fold_summary(model: str, outcome) -> str:
    selection, result = outcome.selection, outcome.heldout_result
    lines = [
        f"{model} outer fold {outcome.outer_fold} (held out {outcome.heldout_build}): "
        f"winner={selection['winner']} epochs={selection['epochs']} "
        f"threshold={selection['threshold']:.3f}",
        f"  held-out mAP@0.5={_fmt(result['map50'])} recall={_fmt(result['recall'])} "
        f"precision={_fmt(result['precision'])}",
    ]
    for inner, flag in sorted(selection["inner_validation_flags"].items()):
        if flag == "unreliable":
            lines.append(
                f"  inner fold {inner}: unreliable validation build (fewer labeled images than "
                "the configured minimum)"
            )
    return "\n".join(lines)


def run_train(
    ctx: typer.Context,
    config: str | None,
    profile: Profile | None,
    model: str,
    fold: int | None,
    dry_run: bool,
    repro: bool,
) -> None:
    from layerwise.detect import protocol
    from layerwise.detect.config import MODELS
    from layerwise.detect.models import make_detector

    cfg = load_detect_ctx_config(ctx, config, profile)
    if model not in MODELS:
        fail(f"unknown model {model!r}; expected one of {', '.join(MODELS)}", EXIT_CONFIG)
    folds = None if fold is None else [fold]
    try:
        if dry_run:
            for job in protocol.plan_jobs(cfg, model, folds, repro):
                typer.echo(f"{'done   ' if job['done'] else 'planned'} {job['label']}")
            return
        summary = protocol.run_protocol(
            cfg,
            model,
            lambda name, candidate: make_detector(cfg, name, candidate),
            folds=folds,
            repro=repro,
            report=lambda stage, label, status: typer.echo(f"{status:<8} {stage:<9} {label}"),
        )
    except tuple(DETECT_ERRORS) as exc:
        fail(str(exc), detect_exit_code_for(exc))
    for note in summary.notes:
        typer.echo(note)
    for outcome in summary.outcomes:
        typer.echo(format_fold_summary(model, outcome))


@app.command()
def train(
    ctx: typer.Context,
    config: ConfigOption = None,
    profile: ProfileOption = None,
    model: ModelOption = "yolo",
    fold: FoldOption = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print planned jobs; write nothing.")
    ] = False,
    repro: Annotated[
        bool, typer.Option("--repro", help="Rerun a finished fold into repro/.")
    ] = False,
) -> None:
    """Run the nested leave-one-build-out protocol for a model."""
    run_train(ctx, config, profile, model, fold, dry_run, repro)


@app.command()
def baseline(
    ctx: typer.Context,
    config: ConfigOption = None,
    profile: ProfileOption = None,
    fold: FoldOption = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print planned jobs; write nothing.")
    ] = False,
    repro: Annotated[
        bool, typer.Option("--repro", help="Rerun a finished fold into repro/.")
    ] = False,
) -> None:
    """Run the non-learned baseline through the same protocol (train --model contrast)."""
    run_train(ctx, config, profile, "contrast", fold, dry_run, repro)


@app.command()
def report(
    ctx: typer.Context, config: ConfigOption = None, profile: ProfileOption = None
) -> None:
    """Compare all finished models and the baseline (comparison.parquet/.md, report.json)."""
    from layerwise.detect import report as detect_report

    cfg = load_detect_ctx_config(ctx, config, profile)
    try:
        result = detect_report.write_report(cfg)
    except tuple(DETECT_ERRORS) as exc:
        fail(str(exc), detect_exit_code_for(exc))
    typer.echo(f"report written to {cfg.reports_dir}")
    typer.echo(result.markdown)


@app.command()
def predict(
    ctx: typer.Context,
    config: ConfigOption = None,
    profile: ProfileOption = None,
    run: Annotated[
        str, typer.Option("--run", help="Model run: <model>/outer_<k> (final model of the fold).")
    ] = "yolo/outer_0",
    build: Annotated[
        str | None, typer.Option("--build", help="Only this build (default: the held-out one).")
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="At most N layers.")] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", help="Override the fold's operating point.")
    ] = None,
) -> None:
    """Write the common per-layer defect representation (detections and defect maps)."""
    from layerwise.detect.models import make_detector
    from layerwise.detect.predict import run_predict

    cfg = load_detect_ctx_config(ctx, config, profile)
    try:
        result = run_predict(
            cfg,
            run,
            lambda name, candidate: make_detector(cfg, name, candidate),
            build=build,
            limit=limit,
            threshold=threshold,
        )
    except tuple(DETECT_ERRORS) as exc:
        fail(str(exc), detect_exit_code_for(exc))
    for image_id in result.unreadable:
        typer.echo(f"warning: unreadable layer {image_id}", err=True)
    typer.echo(
        f"{result.layers} layer(s), {result.detections} detection(s) at threshold "
        f"{result.threshold:.3f} written to {result.directory}"
    )


@app.command()
def benchmark(
    ctx: typer.Context,
    config: ConfigOption = None,
    profile: ProfileOption = None,
    run: Annotated[
        str, typer.Option("--run", help="Model run: <model>/outer_<k> (final model of the fold).")
    ] = "yolo/outer_0",
    runs: Annotated[int | None, typer.Option("--runs", help="Timed iterations.")] = None,
) -> None:
    """Measure per-layer latency on this hardware against the recoat-cycle budget."""
    from layerwise.detect.latency import run_benchmark
    from layerwise.detect.models import make_detector

    cfg = load_detect_ctx_config(ctx, config, profile)
    try:
        result = run_benchmark(
            cfg, run, lambda name, candidate: make_detector(cfg, name, candidate), runs=runs
        )
    except tuple(DETECT_ERRORS) as exc:
        fail(str(exc), detect_exit_code_for(exc))
    rec = result.record
    typer.echo(
        f"{rec['model']} on {rec['hardware']} ({rec['device_name']}): "
        f"p50 {rec['p50_ms']:.1f} ms, p95 {rec['p95_ms']:.1f} ms over {rec['runs']} runs; "
        f"budget {rec['budget_ms']:.0f} ms "
        f"(recoat cycle {result.recoat['shortest_median_s']:.1f} s)"
    )
    if not rec["within_budget"]:
        typer.echo("over the latency budget: not eligible as the default model", err=True)
        raise typer.Exit(EXIT_OVER_BUDGET)

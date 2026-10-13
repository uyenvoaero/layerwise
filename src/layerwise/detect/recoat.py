"""Recoat cycle from the PB layer timestamps: the time budget for per-layer inference."""

from pathlib import Path
from typing import Any

from layerwise.detect.config import DetectConfig
from layerwise.errors import ConfigError
from layerwise.ingest.sqlfiles import load_sql, open_tables
from layerwise.manifest import write_json

SQL_DIR = Path(__file__).parent / "sql"
METHOD = "median time between consecutive PB layers with layer index difference 1"


def recoat_cycle(processed_dir: Path, budget_fraction: float) -> dict[str, Any]:
    """Per-build median layer gap and the budget (shortest median times `budget_fraction`)."""
    con = open_tables(processed_dir, ["images"])
    try:
        rows = con.execute(load_sql("recoat_cycle.sql", "gaps", SQL_DIR)).fetchall()
    finally:
        con.close()
    if not rows:
        raise ConfigError("no consecutive PB layers with timestamps: no recoat cycle to derive")
    per_build = {
        build: {"median_gap_s": float(median), "gaps_used": int(used)}
        for build, median, used in rows
    }
    shortest = min(entry["median_gap_s"] for entry in per_build.values())
    return {
        "per_build": per_build,
        "shortest_median_s": shortest,
        "budget_fraction": budget_fraction,
        "budget_s": shortest * budget_fraction,
        "method": METHOD,
    }


def write_recoat_cycle(cfg: DetectConfig) -> dict[str, Any]:
    result = recoat_cycle(cfg.ingest.processed_dir, float(cfg.latency["budget_fraction"]))
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.reports_dir / "recoat_cycle.json", result)
    return result

# Contract: CLI (feature 002)

Extends the `layerwise` Typer app from feature 001 ([../../001-peregrine-data-ingest/contracts/cli.md](../../001-peregrine-data-ingest/contracts/cli.md)).
All paths and parameters come from the YAML profile in `configs/detect/`; options only select the
profile, the config file and the work to do. `DATA_ROOT` is read from the environment. The detect
commands read the feature 001 outputs, so `layerwise ingest` must have run with the matching ingest
profile (the detect config names it).

Global options (also accepted after the command, as in feature 001): `--config PATH` (default
`configs/detect/base.yaml` for the commands below), `--profile NAME` (`local` | `aws` | `fixture`;
default `local`). `local` and `fixture` use the committed fixture settings on a CPU; `aws` uses full
data and CUDA. The commands require the optional `vision` dependency group; without it they exit `2`
with an install hint.

## `layerwise train`

Run the nested leave-one-build-out protocol for a model.

| Option | Meaning |
|---|---|
| `--model NAME` | `yolo` (default), `frcnn` or `contrast` (the baseline has no training; it runs the threshold step) |
| `--fold N` | Run only outer fold `N` (0..2); default all folds |
| `--dry-run` | Print the planned jobs (candidates, inner folds, final) and which are already done; write nothing |
| `--repro` | Rerun the selected fold(s) from their `selection.json` into `repro/` (ignoring finished jobs, same seed) for the reproduction check; requires a finished fold |

Behaviour: runs the fold-overlap check first; for each outer fold runs inner jobs, writes
`selection.json`, trains the final model, evaluates once on the held-out build. Finished jobs are
skipped; an interrupted job resumes from its last checkpoint. Prints per fold the winner, epochs,
threshold, held-out mAP@0.5, recall and precision, and flags unreliable inner validation builds.

Exit codes: `0` success or nothing to do; `1` fold-overlap check failed (names fold and build);
`2` configuration error (missing `DATA_ROOT`, missing ingest output, missing `vision` extras, bad
YAML); `3` a labeled image listed in the fold table is missing or unreadable above the configured
fraction (same rule as feature 001's `failure_threshold`).

## `layerwise baseline`

Shortcut for `train --model contrast`; runs the non-learned baseline through the same protocol and
metrics on all outer folds (or one with `--fold N`). Same options and exit codes.

## `layerwise predict`

Write the common defect representation for layers.

| Option | Meaning |
|---|---|
| `--run NAME` | Model run to use: `<model>/outer_<k>` (final model of that fold) |
| `--build ID` | Only layers of this build (default: the held-out build of the fold) |
| `--limit N` | Process at most N layers (sorted by layer index) |
| `--threshold X` | Override the fold's operating point (recorded in `layers.parquet`) |

Output: `$DATA_ROOT/predictions/<model>/outer_<k>/` per [defect-representation.md](defect-representation.md).
Exit codes: `0`; `2` configuration error or run not found; `3` unreadable layers above the failure threshold.

## `layerwise benchmark`

Per-layer latency on the current hardware.

| Option | Meaning |
|---|---|
| `--run NAME` | As for `predict` |
| `--runs N` | Timed iterations (default from config) |

Writes the latency record (see [run-outputs.md](run-outputs.md)), merging with records from other
hardware so CPU and GPU results sit in one file. Also writes the recoat-cycle record. Exit `0`;
`2` configuration error; `4` model over the latency budget on this hardware (the report is still
written; the model is marked `eligible_default: false`).

## `layerwise report`

Build the comparison of all finished models and the baseline.

Writes `comparison.parquet`, `comparison.md` and `report.json` under `$DATA_ROOT/reports/detect/`:
per held-out build and pooled mAP@0.5, defect recall, precision, F1, recall by size, the label source,
the counts behind each figure, the flags (unreliable inner validation, builds without labeled boxes, PB images with an ingest
`size_flag`), one verdict per model versus the baseline (`better`, `equal`, `worse`), the
`default_model` (best eligible model by pooled mAP@0.5, see data-model.md) and, for folds rerun with
`train --repro`, the reproduction check against the 0.02 tolerance. A model that does not beat the
baseline on a metric is stated as such.

Partial results: the report is written from the outer folds and models that have finished. Pooled rows
cover the finished folds only, carry `partial = true` and are marked partial in `comparison.md`; the
missing folds and models are listed. Exit `0` when the baseline and at least one model have a finished
fold; exit `2` when no results exist or the baseline is missing (names what is missing).

## Makefile targets

`train`, `baseline`, `predict`, `benchmark`, `report` call the commands above. `smoke` copies the
fixture under a temporary `DATA_ROOT`, ingests it with the ingest `fixture` profile and runs one outer
fold with the `local` profile. `sync-up` and `sync-down` run `aws s3 sync` between
`$DATA_ROOT/{runs,predictions,reports}` and `$S3_URI`; the library itself has no AWS code.

## Stability

Command names, option names and exit codes above are covered by contract tests (`tests/contract/`).
Output formats: [defect-representation.md](defect-representation.md) and [run-outputs.md](run-outputs.md).

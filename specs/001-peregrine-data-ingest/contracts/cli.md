# Contract: CLI

Entry point: `layerwise` (Typer). All paths and parameters come from the YAML profile; options
only select the profile and override the config file location. `DATA_ROOT` is read from the
environment.

Global options: `--config PATH` (default `configs/ingest/base.yaml`), `--profile NAME`
(`local` | `aws` | `fixture`; default `local`). The `fixture` profile is for tests: it reads
`tests/fixtures/` content copied under a temporary `DATA_ROOT` and uses scaled expected image sizes.

## `layerwise ingest`

Scan the configured dataset and write the Parquet tables, report and manifest to
`$DATA_ROOT/processed/<dataset>/`.

| Option | Meaning |
|---|---|
| `--force` | Rewrite even if inputs, config and mapping version are unchanged |
| `--dry-run` | Scan and print the report counts; write nothing |

Behaviour:
- Unchanged inputs (same input hash, config hash, mapping version) and no `--force`: prints
  "up to date" and exits 0 without rewriting.
- Output is written to a temporary directory and renamed into place on success.
- Prints a short summary (images, labeled/unlabeled per modality, pairs, boxes, folds).

Exit codes: `0` success or up to date; `2` configuration error (missing `DATA_ROOT`, source
folder not found, bad YAML); `3` unmapped source class (names dataset and class); `4` input
errors above the configured `failure_threshold` (maximum fraction of unreadable or corrupt files,
default `0.05`); below it, unreadable files are recorded in `failed_files` in the report and the run
continues.

## `layerwise check-splits`

Validate `folds.parquet` in the processed directory.

Fails (exit `1`) if any build has more than one role within an outer or inner fold, or if the test
build of an outer fold appears in any of its inner folds; prints the offending fold and build.
Exit `0` and prints per-role image, labeled-image and box counts when valid.

## `layerwise make-fixture`

Build the committed smoke-test fixture from the full dataset into `tests/fixtures/<dataset>/`
(per build: 20 labeled PB + 20 labeled OT, downscaled, boxes rescaled, `ATTRIBUTION.md`).
Exit `0` on success; exit `1` if the result exceeds the 5 MB limit or a build has fewer than 20
labeled images of a modality.

## Stability

Command names, option names and exit codes above are covered by contract tests
(`tests/contract/`). Parquet schemas: see [parquet-schemas.md](parquet-schemas.md).

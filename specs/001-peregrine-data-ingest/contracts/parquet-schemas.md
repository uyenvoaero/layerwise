# Contract: Output tables

Directory: `$DATA_ROOT/processed/<dataset>/` (for Aalto: `processed/aalto/`). Rows sorted by the
key shown; column order as listed; no timestamps. Types are Arrow types.

## images.parquet (key: image_id)

| Column | Type | Null |
|---|---|---|
| image_id | string | no |
| dataset | string | no |
| build_id | string | no |
| modality | string (`PB`/`OT`) | no |
| layer_index | int32 | no |
| timestamp | string | PB: no, OT: yes |
| z_mm | float64 | PB: yes, OT: no |
| path | string (relative to DATA_ROOT) | no |
| width | int32 | no |
| height | int32 | no |
| mode | string | no |
| size_flag | bool | no |
| label_status | string (`labeled`/`labeled_no_defects`/`unlabeled`) | no |
| use_for_detection | bool | no |

## boxes.parquet (key: box_id)

| Column | Type | Null |
|---|---|---|
| box_id | string | no |
| image_id | string | no |
| xmin, ymin, xmax, ymax | int32 | no |
| source_class | string | no |
| phase_category | string | no |
| fine_class | string | yes (always null for Aalto) |
| clipped | bool | no |

Invariants: `0 <= xmin < xmax <= width` and `0 <= ymin < ymax <= height` of the parent image.

## class_mapping.parquet (key: dataset, source_class)

| Column | Type | Null |
|---|---|---|
| dataset | string | no |
| source_class | string | no |
| phase_category | string | no |
| fine_class | string | yes |
| mapping_version | string | no |

## layer_pairs.parquet (key: build_id, layer_index)

| Column | Type | Null |
|---|---|---|
| build_id | string | no |
| layer_index | int32 | no |
| pb_image_id | string | yes |
| ot_image_id | string | yes |
| pair_status | string (`paired`/`pb_only`/`ot_only`) | no |

## folds.parquet (key: level, outer_fold, inner_fold, build_id)

| Column | Type | Null |
|---|---|---|
| level | string (`outer`/`inner`) | no |
| outer_fold | int32 | no |
| inner_fold | int32 | outer rows: yes |
| build_id | string | no |
| role | string (`test`/`train`/`validation`) | no |

## manifest.json and ingest_report.json

JSON, UTF-8, keys sorted, two-space indent, trailing newline, no timestamps.
`manifest.json` keys: `dataset`, `source` (`name`, `doi`, `license`), `code_version` (package
version), `adapter_version`, `input_hash`, `config_hash`, `config` (resolved parameters, paths
relative to DATA_ROOT), `mapping_version`, `seed` (null), `outputs` (name -> row count), `metrics`
(totals: `images`, `labeled`, `labeled_no_defects`, `unlabeled`, `boxes`, `paired_layers`, `folds`).
`ingest_report.json` keys: `per_build` (build -> modality -> counts), `boxes` (per source class:
`ingested`, `clipped`, `dropped`), `unparsed_files`, `unknown_build_files`, `orphan_xml`,
`duplicates`, `failed_files`.

## Class mapping config (`configs/class_mapping/aalto.yaml`)

```yaml
version: "1"
dataset: aalto
classes:
  defects:    {phase_category: recoating_phase_anomaly, fine_class: null}
  overheated: {phase_category: melting_phase_anomaly,   fine_class: null}
```

Validation: every class seen in XML is listed; no entry has `fine_class: porosity`.

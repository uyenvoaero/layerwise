# Contract: Run outputs, manifests and reports

All paths below are under `$DATA_ROOT`. JSON files are UTF-8, keys sorted, two-space indent, trailing
newline. Timing lives in a separate `timing.json` so manifests of identical inputs compare equal.

## Run layout

```text
runs/detect/<model>/outer_<k>/
├── inner_<j>/<candidate_id>/        # train on one build, validate on the other
│   ├── weights/last.*, best.*       # checkpoints (never committed)
│   ├── result.json                  # best epoch, validation metrics at image level
│   ├── manifest.json
│   └── timing.json
├── selection.json                   # winner, epochs, threshold, inner scores and std, flags
├── final/                           # train on both training builds
│   ├── weights/last.*
│   ├── result.json, manifest.json, timing.json
├── heldout_result.json              # metrics on the held-out build at the selected threshold
├── heldout_detections.parquet       # all predictions above the evaluation floor
└── repro/                           # rerun of the fold (train --repro): same files, for the reproduction check
staging/<model>/outer_<k>/<job>/     # tile JPEGs, label files, dataset YAML (rebuildable)
predictions/<model>/outer_<k>/       # defect-representation.md
reports/detect/                      # comparison.parquet, comparison.md, report.json, latency.json, recoat_cycle.json
```

## manifest.json (per job)

Keys: `job` (model, outer_fold, kind, inner_fold, candidate_id), `data_subset_hash` (ingest
`input_hash` + fold definition + image IDs used), `config` (resolved parameters, paths relative to
DATA_ROOT), `config_hash`, `code_version`, `seed`, `device` (type, name), `libraries` (torch,
torchvision, ultralytics, opencv versions), `train_builds`, `validation_builds`, `test_builds`,
`metrics`, `label_source` (`aalto_manual_pb_boxes`), `tracking_run_id`.

## selection.json

`winner`, `epochs`, `threshold`, `inner_scores` (candidate -> inner fold -> `map50`, `best_epoch`,
`labeled_images`, `labeled_boxes`), `inner_std`, `inner_validation_flags` (inner fold -> `ok` |
`unreliable`; `unreliable` means fewer than 150 labeled images in the validation build).

## heldout_result.json

`held_out_build`, `threshold`, `map50` (number or null with `map50_reason: not_defined`), `precision`,
`recall`, `f1`, `recall_by_size` (bucket -> `{recall, boxes}`), `counts` (`images`, `boxes`,
`predictions`, `matches`, `false_alarms`, `misses`), `matching` (`one_to_one`, `iou: 0.5`),
`label_source`, `model_id`.

## heldout_detections.parquet (key: detection_id)

`detection_id`, `image_id`, `xmin`, `ymin`, `xmax`, `ymax`, `confidence`, `outer_fold`. Rows sorted by key.

## comparison.parquet (key: model, scope, metric)

| Column | Type | Null |
|---|---|---|
| model | string | no |
| scope | string (`build:<id>` or `pooled`) | no |
| metric | string (`map50`, `recall`, `precision`, `f1`, `recall_small`, `recall_medium`, `recall_large`) | no |
| value | float64 | yes (null = not defined) |
| labeled_images | int32 | no |
| labeled_boxes | int32 | no |
| threshold | float64 | yes (null for `map50`) |
| label_source | string | no |
| verdict_vs_baseline | string (`better`/`equal`/`worse`/`not_applicable`) | no |
| partial | bool (pooled scope with fewer than 3 finished outer folds) | no |

`pooled` is computed on the union of the three held-out sets, each build predicted by the model of its
own fold; precision, recall and F1 use each fold's own threshold. `equal` means within the
reproduction tolerance (0.02 absolute). With fewer than 3 finished folds the pooled rows cover the
finished folds only and have `partial = true`; `comparison.md` says so next to every pooled figure.

## report.json

Keys: `models`, `label_source`, `folds_finished` (model -> list of outer folds), `partial` (true if
any model has fewer than 3 finished folds), `inner_validation_flags` (outer fold -> inner fold ->
`ok`/`unreliable`), `builds_without_labeled_boxes`, `size_flagged_images` (build -> count of PB images
whose ingest `size_flag` is true), `default_model` (model name or null, with `default_model_reason`),
`reproduction` (per model and outer fold: `map50_original`, `map50_rerun`, `recall_original`,
`recall_rerun`, `max_abs_diff`, `tolerance` 0.02, `reproduced`; present only for folds that have a
`repro/` directory).

## latency.json

List of records, sorted by `(model, hardware, runtime)`: `model`, `hardware` (`cpu`/`cuda`),
`device_name`, `runtime` (`pytorch`), `precision` (`fp32`/`fp16`), `batch_size` (layers per call),
`tile_batch_size`, `input_resolution` (`[W, H]`), `tile_size`, `runs`, `warmup`, `p50_ms`, `p95_ms`,
`budget_ms`, `within_budget`, `eligible_default`, `excludes` (`["jpeg_decode"]`).

## recoat_cycle.json

`per_build` (build -> `median_gap_s`, `gaps_used`), `shortest_median_s`, `budget_fraction`,
`budget_s`, `method` (median time between consecutive PB layers with layer index difference 1).

## Rules

- Nothing under `runs/`, `staging/`, `predictions/` or `reports/` is committed.
- Every number in `comparison.md` comes from `comparison.parquet`; the markdown is generated, not edited.
- Reproduction check: `train --fold k --repro` reruns an outer fold into `repro/`; `report` compares its
  `map50` and `recall` with `heldout_result.json` and they must be within 0.02 absolute.

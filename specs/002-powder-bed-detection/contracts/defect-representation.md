# Contract: Common per-layer defect representation

`format_version: "1"`. The same files are written by Phase A (box detectors, baseline) and Phase B
(segmentation); consumers (features 003 to 005) must not depend on which model produced them.

Directory: `$DATA_ROOT/predictions/<model>/outer_<k>/`. Rows sorted by the key shown; column order as
listed; types are Arrow types; no timestamps, so identical inputs give identical files.

## layers.parquet (key: image_id)

| Column | Type | Null |
|---|---|---|
| image_id | string | no |
| build_id | string | no |
| layer_index | int32 | no |
| size_flag | bool (the ingest flag: image size differs from the expected size for its modality) | no |
| status | string (`processed`) | no |
| n_detections | int32 | no |
| map_path | string (relative to DATA_ROOT) | no |
| model_id | string | no |
| manifest_path | string (relative to DATA_ROOT; the job manifest of the model) | no |
| threshold | float64 | no |
| format_version | string | no |

A layer with no detections at the threshold has `status = processed` and `n_detections = 0`. A layer
that was not processed has no row at all, so "no detections" and "not processed" are never confused.

## detections.parquet (key: detection_id)

| Column | Type | Null |
|---|---|---|
| detection_id | string (`<image_id>#<n>`, n in descending confidence) | no |
| image_id | string | no |
| xmin, ymin, xmax, ymax | int32 (image pixels, clipped to the image) | no |
| source_class | string (`defects`) | no |
| phase_category | string (`recoating_phase_anomaly`) | no |
| confidence | float32, in [0, 1] | no |

Invariant: `0 <= xmin < xmax <= width` and `0 <= ymin < ymax <= height` of the parent image. Only
detections with `confidence >= threshold` of the run are stored. Phase B segmentation outputs may
leave `detections.parquet` empty or fill it with connected-component boxes; the map is authoritative
for them.

## Defect map (one file per layer)

Path: `maps/<image_id>.npz` under the prediction directory, compressed NumPy archive with:

| Key | Content |
|---|---|
| `defect_map` | float16 array, shape `(C, H, W)`, `H` and `W` equal to the layer image height and width, values in [0, 1] |
| `categories` | string array of length `C`, the phase category of each channel, in a fixed order (Phase A: `["recoating_phase_anomaly"]`) |
| `format_version` | string `"1"` |

Channels are indexed by phase category; a category with no detection is an all-zero channel (not
omitted). For box models each pixel holds the highest confidence among the boxes of that category
covering it, and 0 elsewhere; the map is a deterministic function of `detections.parquet`, so tests
rebuild it and compare. Consumers needing instances use `detections.parquet`; consumers needing
areas (feature 003) use the map thresholded at their own value.

## Rules

- Map width and height equal the layer image size (100% of processed layers, SC-005).
- Every stored output links to the model's job manifest through `manifest_path`.
- Category names come from the feature 001 class mapping; no category is named porosity.
- Adding channels for new phase categories (for example `melting_phase_anomaly` from an OT model)
  is a minor change and keeps `format_version`; changing the array layout or value range is a major
  change and bumps it.

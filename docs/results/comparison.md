# Powder bed anomaly detection: comparison

Label source: Aalto manual PB boxes (recoating-phase anomaly; not porosity). Matching: one-to-one, IoU 0.5. A verdict is `equal` within 0.02 absolute. Generated from `comparison.parquet`; do not edit.

- contrast: finished folds 0, 1, 2
- frcnn: finished folds 0, 1, 2
- yolo: finished folds 0, 1, 2

## build:SI383820211201123521

| model | images | boxes | map50 | recall | precision | f1 | recall_small | recall_medium | recall_large |
|---|---|---|---|---|---|---|---|---|---|
| contrast | 407 | 572 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| frcnn | 407 | 572 | 0.214 (better) | 0.082 (better) | 0.276 (better) | 0.127 (better) | 0.045 (better) | 0.118 (better) | 0.061 (better) |
| yolo | 407 | 572 | 0.158 (better) | 0.122 (better) | 0.357 (better) | 0.182 (better) | 0.106 (better) | 0.224 (better) | 0.043 (better) |

## build:SI383820240226095904

| model | images | boxes | map50 | recall | precision | f1 | recall_small | recall_medium | recall_large |
|---|---|---|---|---|---|---|---|---|---|
| contrast | 127 | 467 | 0.025 | 0.298 | 0.079 | 0.125 | 0.237 | 0.457 | not defined |
| frcnn | 127 | 467 | 0.298 (better) | 0.478 (better) | 0.385 (better) | 0.426 (better) | 0.432 (better) | 0.597 (better) | not defined (not_applicable) |
| yolo | 127 | 467 | 0.446 (better) | 0.619 (better) | 0.538 (better) | 0.576 (better) | 0.530 (better) | 0.853 (better) | not defined (not_applicable) |

## build:SI383820240318120348

| model | images | boxes | map50 | recall | precision | f1 | recall_small | recall_medium | recall_large |
|---|---|---|---|---|---|---|---|---|---|
| contrast | 996 | 3961 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| frcnn | 996 | 3961 | 0.270 (better) | 0.589 (better) | 0.202 (better) | 0.301 (better) | 0.238 (better) | 0.464 (better) | 0.773 (better) |
| yolo | 996 | 3961 | 0.269 (better) | 0.656 (better) | 0.119 (better) | 0.201 (better) | 0.282 (better) | 0.562 (better) | 0.831 (better) |

## pooled

| model | images | boxes | map50 | recall | precision | f1 | recall_small | recall_medium | recall_large |
|---|---|---|---|---|---|---|---|---|---|
| contrast | 1530 | 5000 | 0.001 | 0.028 | 0.015 | 0.019 | 0.071 | 0.040 | 0.000 |
| frcnn | 1530 | 5000 | 0.257 (better) | 0.521 (better) | 0.212 (better) | 0.301 (better) | 0.285 (better) | 0.422 (better) | 0.691 (better) |
| yolo | 1530 | 5000 | 0.270 (better) | 0.592 (better) | 0.131 (better) | 0.214 (better) | 0.346 (better) | 0.535 (better) | 0.740 (better) |

## Against the baseline

- frcnn beats the baseline on pooled mAP@0.5.
- frcnn beats the baseline on pooled recall.
- frcnn beats the baseline on pooled precision.
- frcnn beats the baseline on pooled F1.
- yolo beats the baseline on pooled mAP@0.5.
- yolo beats the baseline on pooled recall.
- yolo beats the baseline on pooled precision.
- yolo beats the baseline on pooled F1.

## Flags

- Outer fold 0, inner fold 1: unreliable validation build (fewer labeled images than the configured minimum).
- Outer fold 2, inner fold 0: unreliable validation build (fewer labeled images than the configured minimum).

Default model for later features: yolo.

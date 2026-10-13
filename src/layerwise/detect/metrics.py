"""Detection metrics over numpy arrays, shared by every model and the baseline.

Matching is one-to-one: per image, predictions in descending confidence take the unmatched labeled
box with the highest IoU of at least `iou`. Unmatched predictions are false alarms, unmatched
labeled boxes are misses. A build without labeled boxes has recall and mAP "not defined" (None).
"""

from dataclasses import dataclass

import numpy as np

from layerwise.detect.tiling import iou_matrix

SIZE_BUCKETS = ("small", "medium", "large")


@dataclass(frozen=True)
class ImageEval:
    """Labeled boxes and predictions of one image, boxes as (N, 4) xmin, ymin, xmax, ymax."""

    image_id: str
    gt: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    """All predictions with their match flag and, per labeled box, the score that matched it."""

    pred_scores: np.ndarray
    pred_tp: np.ndarray
    gt_scores: np.ndarray  # score of the matching prediction, -inf when no prediction matched
    gt_areas: np.ndarray
    n_images: int


def match_images(images: list[ImageEval], iou: float = 0.5) -> MatchResult:
    pred_scores: list[np.ndarray] = []
    pred_tp: list[np.ndarray] = []
    gt_scores: list[np.ndarray] = []
    gt_areas: list[np.ndarray] = []
    for item in images:
        gt = np.asarray(item.gt, float).reshape(-1, 4)
        boxes = np.asarray(item.boxes, float).reshape(-1, 4)
        scores = np.asarray(item.scores, float).reshape(-1)
        order = np.argsort(-scores, kind="stable")
        boxes, scores = boxes[order], scores[order]
        overlaps = iou_matrix(boxes, gt)
        taken = np.zeros(len(gt), bool)
        matched_score = np.full(len(gt), -np.inf)
        tp = np.zeros(len(boxes), bool)
        for i in range(len(boxes)):
            if len(gt) == 0:
                break
            candidate = np.where(taken, -1.0, overlaps[i])
            j = int(np.argmax(candidate))
            if candidate[j] >= iou:
                taken[j] = True
                matched_score[j] = scores[i]
                tp[i] = True
        pred_scores.append(scores)
        pred_tp.append(tp)
        gt_scores.append(matched_score)
        gt_areas.append((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]))
    return MatchResult(
        pred_scores=_cat(pred_scores),
        pred_tp=_cat(pred_tp, bool),
        gt_scores=_cat(gt_scores),
        gt_areas=_cat(gt_areas),
        n_images=len(images),
    )


def _cat(parts: list[np.ndarray], dtype=float) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)


def concat_matches(matches: list[MatchResult]) -> MatchResult:
    return MatchResult(
        pred_scores=_cat([m.pred_scores for m in matches]),
        pred_tp=_cat([m.pred_tp for m in matches], bool),
        gt_scores=_cat([m.gt_scores for m in matches]),
        gt_areas=_cat([m.gt_areas for m in matches]),
        n_images=sum(m.n_images for m in matches),
    )


def average_precision(match: MatchResult) -> float | None:
    """Area under the precision-recall curve (all-point interpolation); None without boxes."""
    n_gt = len(match.gt_scores)
    if n_gt == 0:
        return None
    if len(match.pred_scores) == 0:
        return 0.0
    order = np.argsort(-match.pred_scores, kind="stable")
    tp = match.pred_tp[order].astype(float)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = np.concatenate([[0.0], cum_tp / n_gt])
    precision = np.concatenate([[1.0], cum_tp / (cum_tp + cum_fp)])
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * envelope[1:]))


def operating_point(match: MatchResult, threshold: float) -> dict:
    """Counts, precision, recall and F1 keeping predictions with confidence >= threshold.

    With labeled boxes but no predictions precision is 0.0; without labeled boxes recall and F1
    are None (not defined).
    """
    kept = match.pred_scores >= threshold
    predictions = int(kept.sum())
    matches = int(match.pred_tp[kept].sum())
    false_alarms = predictions - matches
    n_gt = len(match.gt_scores)
    misses = n_gt - matches
    precision = matches / predictions if predictions else 0.0
    recall = matches / n_gt if n_gt else None
    if recall is None:
        f1 = None
    elif precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "images": match.n_images,
        "boxes": n_gt,
        "predictions": predictions,
        "matches": matches,
        "false_alarms": false_alarms,
        "misses": misses,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def best_f1_threshold(match: MatchResult) -> float | None:
    """Confidence threshold that maximises F1; ties go to the higher threshold."""
    if len(match.gt_scores) == 0 or len(match.pred_scores) == 0:
        return None
    best_threshold, best_f1 = None, -1.0
    for threshold in sorted(set(match.pred_scores.tolist()), reverse=True):
        f1 = operating_point(match, threshold)["f1"]
        if f1 > best_f1 + 1e-12:
            best_threshold, best_f1 = threshold, f1
    return best_threshold


def size_bucket(areas: np.ndarray, buckets: dict[str, int]) -> np.ndarray:
    """Bucket name per area: small < small_max_area <= medium < medium_max_area <= large."""
    names = np.where(
        areas < buckets["small_max_area"],
        "small",
        np.where(areas < buckets["medium_max_area"], "medium", "large"),
    )
    return names


def recall_by_size(match: MatchResult, threshold: float, buckets: dict[str, int]) -> dict:
    names = size_bucket(match.gt_areas, buckets)
    out = {}
    for bucket in SIZE_BUCKETS:
        in_bucket = names == bucket
        n = int(in_bucket.sum())
        hits = int((match.gt_scores[in_bucket] >= threshold).sum())
        out[bucket] = {"recall": hits / n if n else None, "boxes": n}
    return out


def pooled_operating_point(matches: list[MatchResult], thresholds: list[float]) -> dict:
    """Counts summed over sets, each set at its own threshold; precision/recall/F1 from the sums."""
    points = [operating_point(m, t) for m, t in zip(matches, thresholds, strict=True)]
    total = {
        k: sum(p[k] for p in points)
        for k in ("images", "boxes", "predictions", "matches", "false_alarms", "misses")
    }
    precision = total["matches"] / total["predictions"] if total["predictions"] else 0.0
    recall = total["matches"] / total["boxes"] if total["boxes"] else None
    if recall is None:
        f1 = None
    elif precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {**total, "precision": precision, "recall": recall, "f1": f1}


def pooled_recall_by_size(
    matches: list[MatchResult], thresholds: list[float], buckets: dict[str, int]
) -> dict:
    hits = dict.fromkeys(SIZE_BUCKETS, 0)
    boxes = dict.fromkeys(SIZE_BUCKETS, 0)
    for match, threshold in zip(matches, thresholds, strict=True):
        names = size_bucket(match.gt_areas, buckets)
        for bucket in SIZE_BUCKETS:
            in_bucket = names == bucket
            boxes[bucket] += int(in_bucket.sum())
            hits[bucket] += int((match.gt_scores[in_bucket] >= threshold).sum())
    return {
        b: {"recall": hits[b] / boxes[b] if boxes[b] else None, "boxes": boxes[b]}
        for b in SIZE_BUCKETS
    }


def within_tolerance(a: float | None, b: float | None, tolerance: float) -> bool:
    """Reproduction check: absolute difference within tolerance; two "not defined" values agree."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tolerance

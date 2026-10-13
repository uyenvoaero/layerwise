import numpy as np
import pytest

from layerwise.detect import metrics
from layerwise.detect.metrics import ImageEval

BUCKETS = {"small_max_area": 1024, "medium_max_area": 9216}


def image(image_id, gt, preds):
    """preds: list of (box, score)."""
    return ImageEval(
        image_id,
        np.array(gt, float).reshape(-1, 4),
        np.array([p[0] for p in preds], float).reshape(-1, 4),
        np.array([p[1] for p in preds], float),
    )


def three_prediction_example():
    # Two labeled boxes; predictions by descending confidence: hit, false alarm, hit.
    return metrics.match_images(
        [
            image(
                "a",
                [[0, 0, 10, 10], [100, 100, 110, 110]],
                [
                    ([0, 0, 10, 10], 0.9),
                    ([50, 50, 60, 60], 0.8),
                    ([100, 100, 110, 110], 0.7),
                ],
            )
        ]
    )


def test_two_predictions_on_one_box_give_one_match_and_one_false_alarm():
    match = metrics.match_images(
        [image("a", [[0, 0, 10, 10]], [([0, 0, 10, 10], 0.9), ([1, 0, 11, 10], 0.8)])]
    )
    point = metrics.operating_point(match, threshold=0.0)
    assert (point["matches"], point["false_alarms"], point["misses"]) == (1, 1, 0)
    assert point["predictions"] == 2


def test_iou_below_half_is_a_miss_and_a_false_alarm():
    match = metrics.match_images([image("a", [[0, 0, 10, 10]], [([6, 0, 16, 10], 0.9)])])
    point = metrics.operating_point(match, threshold=0.0)  # IoU 4/16 = 0.25
    assert (point["matches"], point["false_alarms"], point["misses"]) == (0, 1, 1)


def test_prediction_takes_the_best_iou_unmatched_box():
    gt = [[0, 0, 10, 10], [2, 0, 12, 10]]
    match = metrics.match_images(
        [image("a", gt, [([2, 0, 12, 10], 0.9), ([2, 0, 12, 10], 0.8)])]
    )
    point = metrics.operating_point(match, threshold=0.0)
    # first prediction takes box 1 (IoU 1.0); the second takes box 0 (IoU 8/12 = 0.67)
    assert (point["matches"], point["false_alarms"], point["misses"]) == (2, 0, 0)


def test_map50_all_point_interpolation_on_three_predictions():
    # precision/recall: (1, 0.5), (0.5, 0.5), (2/3, 1); envelope gives 1 * 0.5 + 2/3 * 0.5
    assert metrics.average_precision(three_prediction_example()) == pytest.approx(5 / 6)


def test_f1_maximising_threshold():
    # F1 at 0.9: 0.667, at 0.8: 0.5, at 0.7: 0.8
    assert metrics.best_f1_threshold(three_prediction_example()) == pytest.approx(0.7)


def test_operating_point_precision_recall_f1():
    point = metrics.operating_point(three_prediction_example(), threshold=0.8)
    assert point["precision"] == pytest.approx(0.5)
    assert point["recall"] == pytest.approx(0.5)
    assert point["f1"] == pytest.approx(0.5)
    assert (point["matches"], point["false_alarms"], point["misses"]) == (1, 1, 1)


def test_recall_by_size_bucket_with_counts():
    small, medium, large = [0, 0, 10, 10], [0, 0, 50, 50], [0, 0, 100, 100]  # 100, 2500, 10000
    match = metrics.match_images(
        [
            image("a", [small], [(small, 0.9)]),
            image("b", [medium, medium], [(medium, 0.9)]),
            image("c", [large], []),
        ]
    )
    by_size = metrics.recall_by_size(match, threshold=0.5, buckets=BUCKETS)
    assert by_size["small"] == {"recall": 1.0, "boxes": 1}
    assert by_size["medium"] == {"recall": 0.5, "boxes": 2}
    assert by_size["large"] == {"recall": 0.0, "boxes": 1}


def test_build_without_labeled_boxes_is_not_defined_never_zero_or_one():
    match = metrics.match_images([image("a", [], [([0, 0, 5, 5], 0.9)])])
    assert metrics.average_precision(match) is None
    point = metrics.operating_point(match, threshold=0.0)
    assert point["recall"] is None and point["f1"] is None
    assert point["false_alarms"] == 1
    assert metrics.best_f1_threshold(match) is None
    assert metrics.recall_by_size(match, 0.0, BUCKETS)["small"] == {"recall": None, "boxes": 0}


def test_no_predictions_gives_zero_recall_and_zero_precision():
    match = metrics.match_images([image("a", [[0, 0, 10, 10]], [])])
    point = metrics.operating_point(match, threshold=0.5)
    assert point["recall"] == 0.0 and point["precision"] == 0.0 and point["f1"] == 0.0
    assert metrics.average_precision(match) == 0.0


def test_pooled_metrics_over_several_held_out_sets():
    one = metrics.match_images([image("a", [[0, 0, 10, 10]], [([0, 0, 10, 10], 0.9)])])
    two = metrics.match_images(
        [image("b", [[0, 0, 10, 10], [20, 20, 30, 30]], [([20, 20, 30, 30], 0.4)])]
    )
    pooled = metrics.concat_matches([one, two])
    # each set uses its own threshold: 0.8 on `one` keeps its hit, 0.3 on `two` keeps its hit
    point = metrics.pooled_operating_point([one, two], thresholds=[0.8, 0.3])
    assert (point["matches"], point["false_alarms"], point["misses"]) == (2, 0, 1)
    assert point["recall"] == pytest.approx(2 / 3)
    assert point["precision"] == 1.0
    assert point["images"] == 2 and point["boxes"] == 3
    assert metrics.average_precision(pooled) == pytest.approx(2 / 3)


def test_pooled_recall_is_not_defined_when_no_set_has_boxes():
    empty = metrics.match_images([image("a", [], [])])
    assert metrics.pooled_operating_point([empty], thresholds=[0.5])["recall"] is None


def test_reproduction_check_uses_absolute_tolerance():
    assert metrics.within_tolerance(0.50, 0.519, 0.02)
    assert not metrics.within_tolerance(0.50, 0.53, 0.02)
    assert not metrics.within_tolerance(None, 0.5, 0.02)
    assert metrics.within_tolerance(None, None, 0.02)

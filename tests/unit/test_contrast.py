import copy
import dataclasses

import numpy as np
import pytest
from tests.conftest import DETECT_CONFIG

pytest.importorskip("cv2")

from layerwise.detect.config import load_detect_config  # noqa: E402
from layerwise.detect.models.contrast import ContrastDetector  # noqa: E402


@pytest.fixture
def cfg(data_root):
    return load_detect_config(DETECT_CONFIG, "fixture")


def detector(cfg, **overrides):
    raw = copy.deepcopy(cfg.raw)
    raw["models"]["contrast"].update(overrides)
    cfg = dataclasses.replace(cfg, raw=raw)
    return ContrastDetector(cfg, cfg.candidates("contrast")[0])


def noisy(shape=(256, 256), seed=0, level=120, sigma=3.0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(level, sigma, shape), 0, 255).astype(np.uint8)


def with_streak(image, contrast, rows=(100, 106), cols=(40, 160)):
    out = image.astype(int).copy()
    out[rows[0] : rows[1], cols[0] : cols[1]] += contrast
    return np.clip(out, 0, 255).astype(np.uint8)


def test_bright_thin_streak_on_noisy_background_is_boxed(cfg):
    boxes, scores = detector(cfg).predict_tiles([with_streak(noisy(), 40)])[0]
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= 42 and y0 <= 101 and x1 >= 158 and y1 >= 105  # covers the streak
    assert x1 - x0 < 130 and y1 - y0 < 15  # and little more
    assert 0.0 < scores[0] <= 1.0


def test_dark_streak_is_found_too(cfg):
    boxes, _ = detector(cfg).predict_tiles([with_streak(noisy(), -40)])[0]
    assert len(boxes) == 1


def test_flat_noisy_image_gives_no_detections(cfg):
    boxes, scores = detector(cfg).predict_tiles([noisy(seed=1), noisy(seed=2)])[0]
    assert boxes.shape == (0, 4) and scores.shape == (0,)


def test_confidence_increases_with_streak_contrast(cfg):
    low = detector(cfg).predict_tiles([with_streak(noisy(), 24)])[0][1]
    high = detector(cfg).predict_tiles([with_streak(noisy(), 36)])[0][1]
    assert len(low) == 1 and len(high) == 1
    assert high[0] > low[0]


def test_components_below_minimum_area_are_dropped(cfg):
    image = with_streak(noisy(), 40, rows=(100, 106), cols=(40, 46))  # 6 x 6 = 36 px
    assert len(detector(cfg, min_area=20).predict_tiles([image])[0][0]) == 1
    assert len(detector(cfg, min_area=60).predict_tiles([image])[0][0]) == 0


def test_parameters_come_from_config_and_nothing_is_trained(cfg):
    det = detector(cfg, threshold_multiple=100.0)
    assert det.trainable is False
    assert len(det.predict_tiles([with_streak(noisy(), 40)])[0][0]) == 0  # threshold not reachable
    with pytest.raises(NotImplementedError):
        det.fit(None, [], None, False)
    det.load(None)  # a no-op for the baseline
    assert det.info()["precision"] == "fp32"

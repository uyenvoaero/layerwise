import math

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torchvision")
torch = pytest.importorskip("torch")

from tests.conftest import DETECT_CONFIG  # noqa: E402

from layerwise.detect.config import load_detect_config  # noqa: E402
from layerwise.detect.models.base import FitJob, TileEntry  # noqa: E402
from layerwise.detect.models.frcnn import FasterRcnnDetector, load_checkpoint  # noqa: E402

TILE = 96


@pytest.fixture
def setup(data_root, tmp_path):
    cfg = load_detect_config(DETECT_CONFIG, "fixture")
    candidate = cfg.candidates("frcnn")[0]
    rng = np.random.default_rng(0)
    path = tmp_path / "img.jpg"
    Image.fromarray(rng.integers(0, 255, (TILE, TILE * 2), dtype=np.uint8), "L").save(path)
    tiles = [
        TileEntry("img", path, (0, 0, TILE, TILE), np.array([[10.0, 10.0, 50.0, 40.0]])),
        TileEntry("img", path, (TILE, 0, 2 * TILE, TILE), np.zeros((0, 4))),
    ]
    return cfg, candidate, tiles


def fit_job(cfg, candidate, tmp_path, epochs):
    tp = cfg.tiling_for(candidate)
    tp = type(tp)(TILE, 16, tp.min_box_visible, tp.negative_tile_ratio, tp.nms_iou)
    return FitJob(
        directory=tmp_path / "job",
        staging_dir=tmp_path / "staging",
        candidate=candidate,
        tiling=tp,
        epochs=epochs,
        patience=5,
        seed=0,
        train={"batch_size": 2, "workers": 0, "amp": False, "device": "cpu"},
        options=cfg.model_options("frcnn"),
    )


def test_one_training_step_returns_a_finite_loss(setup, tmp_path):
    cfg, candidate, tiles = setup
    detector = FasterRcnnDetector(cfg, candidate, tile_size=TILE)
    model = detector.new_model()
    optimizer = detector.make_optimizer(model, cfg.model_options("frcnn"))
    loss = detector.train_step(model, optimizer, tiles)
    assert math.isfinite(loss)


def test_predict_tiles_returns_boxes_with_confidences_in_unit_range(setup, tmp_path):
    cfg, candidate, tiles = setup
    detector = FasterRcnnDetector(cfg, candidate, tile_size=TILE)
    detector.model = detector.new_model()
    detector.model.roi_heads.score_thresh = 0.0
    out = detector.predict_tiles([np.full((TILE, TILE), 100, np.uint8)])
    boxes, scores = out[0]
    assert boxes.shape[1:] == (4,) and len(boxes) == len(scores)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_checkpoint_each_epoch_restores_optimizer_state_and_resumes(setup, tmp_path):
    cfg, candidate, tiles = setup
    detector = FasterRcnnDetector(cfg, candidate, tile_size=TILE)
    job = fit_job(cfg, candidate, tmp_path, epochs=1)
    result = detector.fit(job, tiles, tiles, resume=True)
    assert result.best_epoch == 1 and result.last_weights.exists()

    checkpoint = load_checkpoint(result.last_weights)
    assert checkpoint["epoch"] == 1
    model = detector.new_model()
    optimizer = detector.make_optimizer(model, cfg.model_options("frcnn"))
    optimizer.load_state_dict(checkpoint["optimizer"])
    assert len(optimizer.state_dict()["state"]) > 0  # AdamW moments came back

    again = FasterRcnnDetector(cfg, candidate, tile_size=TILE)
    longer = again.fit(fit_job(cfg, candidate, tmp_path, epochs=2), tiles, tiles, resume=True)
    assert longer.epochs_run == 2 and len(longer.history) == 2  # epoch 1 kept, only epoch 2 ran

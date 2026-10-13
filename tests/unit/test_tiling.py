import numpy as np
import pytest

from layerwise.detect import tiling


def covered(width, height, grid):
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in grid:
        mask[y0:y1, x0:x1] = True
    return mask


def test_grid_covers_every_pixel_and_aligns_to_edge():
    grid = tiling.tile_grid(1280, 1024, tile_size=640, overlap=160)
    assert covered(1280, 1024, grid).all()
    assert max(t[2] for t in grid) == 1280 and max(t[3] for t in grid) == 1024
    assert all(t[2] - t[0] == 640 and t[3] - t[1] == 640 for t in grid)
    xs = sorted({t[0] for t in grid})
    assert xs == [0, 480, 640]  # stride 480, last tile aligned to the edge


def test_grid_for_image_smaller_than_tile():
    assert tiling.tile_grid(320, 256, tile_size=256, overlap=64) == [
        (0, 0, 256, 256),
        (64, 0, 320, 256),
    ]
    assert tiling.tile_grid(100, 80, tile_size=256, overlap=64) == [(0, 0, 100, 80)]


def test_overlap_must_be_smaller_than_tile():
    with pytest.raises(ValueError):
        tiling.tile_grid(100, 100, tile_size=64, overlap=64)


def test_crop_returns_tile_pixels():
    image = np.arange(10 * 12, dtype=np.uint8).reshape(10, 12)
    assert (tiling.crop(image, (2, 3, 6, 8)) == image[3:8, 2:6]).all()


def test_box_round_trip_through_tiles_and_stitching():
    grid = tiling.tile_grid(1280, 1024, tile_size=640, overlap=160)
    boxes = np.array(
        [[10, 20, 300, 80], [500, 100, 620, 400], [700, 700, 1000, 760], [1200, 900, 1270, 1000]],
        dtype=float,
    )  # each box fits one tile
    pieces = []
    for tile in grid:
        kept, _, visible = tiling.clip_boxes_to_tile(boxes, tile, min_visible=0.6)
        if len(kept):
            offset = np.array([tile[0], tile[1], tile[0], tile[1]])
            pieces.append((kept + offset, visible))  # visible fraction acts as the score
    merged, _ = tiling.stitch(pieces, nms_iou=0.5)
    assert len(merged) == len(boxes)
    for box in boxes:
        assert np.abs(merged - box).max(axis=1).min() <= 1


def test_training_tiles_keep_box_only_with_enough_visible_area():
    tile = (0, 0, 100, 100)
    boxes = np.array([[10, 10, 50, 50], [80, 10, 140, 50], [60, 10, 120, 50]], dtype=float)
    kept, idx, visible = tiling.clip_boxes_to_tile(boxes, tile, min_visible=0.6)
    # box 1 has 20 of 60 px inside (0.33) and is dropped; box 2 has 40 of 60 (0.67) and is clipped
    assert idx.tolist() == [0, 2]
    assert kept.tolist() == [[10, 10, 50, 50], [60, 10, 100, 50]]
    assert visible.tolist() == pytest.approx([1.0, 40 / 60])


def test_negative_tiles_are_seeded_and_capped_by_ratio():
    candidates = [(i * 10, 0, i * 10 + 10, 10) for i in range(40)]
    first = tiling.sample_negative_tiles(candidates, n_positive=8, ratio=0.25, seed=3)
    again = tiling.sample_negative_tiles(candidates, n_positive=8, ratio=0.25, seed=3)
    other = tiling.sample_negative_tiles(candidates, n_positive=8, ratio=0.25, seed=4)
    assert len(first) == 2 and first == again and first != other
    assert tiling.sample_negative_tiles(candidates[:1], n_positive=8, ratio=0.25, seed=3) == (
        candidates[:1]
    )
    assert tiling.sample_negative_tiles(candidates, n_positive=0, ratio=0.25, seed=3) == []


def test_stitching_merges_duplicates_in_overlaps_with_nms():
    a = (np.array([[100, 100, 200, 150]], float), np.array([0.9]))
    b = (np.array([[102, 101, 201, 151]], float), np.array([0.7]))
    far = (np.array([[500, 500, 560, 540]], float), np.array([0.5]))
    boxes, scores = tiling.stitch([a, b, far], nms_iou=0.5)
    assert boxes.shape == (2, 4)
    assert scores.tolist() == [0.9, 0.5]  # descending, the weaker duplicate is dropped


def test_stitch_of_nothing_is_empty():
    boxes, scores = tiling.stitch([], nms_iou=0.5)
    assert boxes.shape == (0, 4) and scores.shape == (0,)

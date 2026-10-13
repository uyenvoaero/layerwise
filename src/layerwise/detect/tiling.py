import random

import numpy as np

Tile = tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels, x1 and y1 exclusive


def _starts(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile, stride))
    starts.append(size - tile)  # last tile aligned to the image edge
    return starts


def tile_grid(width: int, height: int, tile_size: int, overlap: int) -> list[Tile]:
    """Overlapping tiles covering every pixel; an image smaller than a tile gives one tile."""
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    stride = tile_size - overlap
    tiles = []
    for y0 in _starts(height, tile_size, stride):
        for x0 in _starts(width, tile_size, stride):
            tiles.append((x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)))
    return tiles


def crop(image: np.ndarray, tile: Tile) -> np.ndarray:
    x0, y0, x1, y1 = tile
    return image[y0:y1, x0:x1]


def clip_boxes_to_tile(
    boxes: np.ndarray, tile: Tile, min_visible: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boxes (N, 4) in image pixels -> (clipped boxes in tile coordinates, indices, visible share).

    A box is kept only when at least `min_visible` of its area lies inside the tile.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x0, y0, x1, y1 = tile
    clipped = np.stack(
        [
            np.clip(boxes[:, 0], x0, x1),
            np.clip(boxes[:, 1], y0, y1),
            np.clip(boxes[:, 2], x0, x1),
            np.clip(boxes[:, 3], y0, y1),
        ],
        axis=1,
    )
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    inside = np.maximum(clipped[:, 2] - clipped[:, 0], 0) * np.maximum(
        clipped[:, 3] - clipped[:, 1], 0
    )
    visible = np.divide(inside, area, out=np.zeros_like(inside), where=area > 0)
    keep = np.flatnonzero((visible >= min_visible) & (inside > 0))
    shifted = clipped[keep] - np.array([x0, y0, x0, y0])
    return shifted, keep, visible[keep]


def sample_negative_tiles(candidates: list, n_positive: int, ratio: float, seed: int) -> list:
    """Seeded choice of box-free tiles (any hashable items), at most `ratio` times the positives."""
    limit = int(round(n_positive * ratio))
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        return list(candidates)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(candidates)), limit))
    return [candidates[i] for i in chosen]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes (N, 4) and (M, 4)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(ix1 - ix0, 0) * np.maximum(iy1 - iy0, 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Indices kept by greedy non-maximum suppression, in descending score order."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    keep: list[int] = []
    for i in order:
        if keep and iou_matrix(boxes[i : i + 1], boxes[keep]).max() > iou:
            continue
        keep.append(int(i))
    return np.array(keep, dtype=int)


def stitch(
    tile_results: list[tuple[np.ndarray, np.ndarray]], nms_iou: float
) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-tile detections that are already in image coordinates with NMS.

    `tile_results` is a list of (boxes (N, 4), scores (N,)); returns boxes and scores sorted by
    descending score.
    """
    if not tile_results:
        return np.zeros((0, 4)), np.zeros((0,))
    boxes = np.concatenate([np.asarray(b, float).reshape(-1, 4) for b, _ in tile_results])
    scores = np.concatenate([np.asarray(s, float).reshape(-1) for _, s in tile_results])
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    keep = nms(boxes, scores, nms_iou)
    return boxes[keep], scores[keep]

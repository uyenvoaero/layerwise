import hashlib
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from layerwise.detect import staging
from layerwise.detect.config import TilingParams
from layerwise.detect.data import LabeledImage

PARAMS = TilingParams(
    tile_size=256, overlap=64, min_box_visible=0.6, negative_tile_ratio=1.0, nms_iou=0.5
)


def make_image(folder: Path, build: str, layer: int, boxes) -> LabeledImage:
    image_id = f"{build}_{layer:05d}"
    path = folder / f"{image_id}.jpg"
    rng = np.random.default_rng(layer)
    Image.fromarray(rng.integers(0, 255, (256, 320), dtype=np.uint8), "L").save(path, "JPEG")
    return LabeledImage(
        image_id=image_id,
        build_id=build,
        layer_index=layer,
        path=path,
        width=320,
        height=256,
        size_flag=False,
        boxes=np.array(boxes, float).reshape(-1, 4),
    )


def images_for(folder: Path):
    folder.mkdir(exist_ok=True)
    return {
        build: [
            make_image(folder, build, 1, [[10, 10, 50, 40]]),
            make_image(folder, build, 2, [[270, 100, 310, 140]]),
            make_image(folder, build, 3, []),  # labeled, no defects
        ]
        for build in ("SIA", "SIB", "SIC")
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(tmp_path, train, validation, seed=0, name="job"):
    train_tiles = staging.build_tile_entries(train, PARAMS, seed)
    val_tiles = staging.build_tile_entries(validation, PARAMS, seed)
    return staging.write_yolo_dataset(tmp_path / name, train_tiles, val_tiles), train_tiles


def test_tiles_labels_and_yaml_are_written_from_train_and_validation_only(tmp_path):
    by_build = images_for(tmp_path / "src")
    staged, _ = stage(tmp_path, by_build["SIA"], by_build["SIB"])
    cfg = yaml.safe_load(staged.yaml_path.read_text(encoding="utf-8"))
    assert cfg["names"] == {0: "defects"}
    train_dir = Path(cfg["path"]) / cfg["train"]
    val_dir = Path(cfg["path"]) / cfg["val"]
    train_names = sorted(p.name for p in train_dir.glob("*.jpg"))
    val_names = sorted(p.name for p in val_dir.glob("*.jpg"))
    assert train_names and val_names
    assert all(n.startswith("SIA_") for n in train_names)
    assert all(n.startswith("SIB_") for n in val_names)  # nothing of SIC (held-out) anywhere
    assert not any("SIC_" in p.name for p in Path(cfg["path"]).rglob("*"))


def test_yolo_label_files_use_class_zero_and_normalised_xywh(tmp_path):
    by_build = images_for(tmp_path / "src")
    staged, entries = stage(tmp_path, by_build["SIA"][:1], by_build["SIB"][:1])
    cfg = yaml.safe_load(staged.yaml_path.read_text(encoding="utf-8"))
    label = next((Path(cfg["path"]) / "labels" / "train").glob("SIA_00001__x0_y0.txt"))
    cls, cx, cy, w, h = label.read_text(encoding="utf-8").split()
    assert cls == "0"
    assert (float(cx), float(cy), float(w), float(h)) == (
        np.float64(30 / 256).round(6),
        np.float64(25 / 256).round(6),
        np.float64(40 / 256).round(6),
        np.float64(30 / 256).round(6),
    )
    jpeg = Image.open(next((Path(cfg["path"]) / "images" / "train").glob("SIA_00001__x0_y0.jpg")))
    assert jpeg.size == (256, 256)
    assert len(entries) >= 1


def test_negative_tiles_have_empty_label_files(tmp_path):
    by_build = images_for(tmp_path / "src")
    staged, entries = stage(tmp_path, by_build["SIA"], by_build["SIB"])
    cfg = yaml.safe_load(staged.yaml_path.read_text(encoding="utf-8"))
    negatives = [e for e in entries if len(e.boxes) == 0]
    assert negatives
    name = f"{negatives[0].image_id}__x{negatives[0].tile[0]}_y{negatives[0].tile[1]}.txt"
    assert (Path(cfg["path"]) / "labels" / "train" / name).read_text(encoding="utf-8") == ""


def test_staging_is_deterministic_for_the_same_seed(tmp_path):
    by_build = images_for(tmp_path / "src")
    a, entries_a = stage(tmp_path, by_build["SIA"], by_build["SIB"], seed=5, name="a")
    b, entries_b = stage(tmp_path, by_build["SIA"], by_build["SIB"], seed=5, name="b")
    assert [(e.image_id, e.tile) for e in entries_a] == [(e.image_id, e.tile) for e in entries_b]
    files_a = sorted(p.relative_to(a.root) for p in a.root.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b.root) for p in b.root.rglob("*") if p.is_file())
    assert files_a == files_b
    for rel in files_a:
        if rel.suffix != ".yaml":  # the dataset YAML holds the staging path
            assert digest(a.root / rel) == digest(b.root / rel)


def test_source_images_are_not_modified(tmp_path):
    by_build = images_for(tmp_path / "src")
    before = {p: digest(p) for p in (tmp_path / "src").glob("*.jpg")}
    stage(tmp_path, by_build["SIA"], by_build["SIB"])
    assert {p: digest(p) for p in (tmp_path / "src").glob("*.jpg")} == before


def test_restaging_rebuilds_missing_files(tmp_path):
    by_build = images_for(tmp_path / "src")
    staged, _ = stage(tmp_path, by_build["SIA"], by_build["SIB"], name="job")
    victim = next(staged.root.rglob("*.jpg"))
    victim.unlink()
    stage(tmp_path, by_build["SIA"], by_build["SIB"], name="job")
    assert victim.exists()


def test_validation_less_dataset_points_val_at_the_training_tiles(tmp_path):
    by_build = images_for(tmp_path / "src")
    train_tiles = staging.build_tile_entries(by_build["SIA"], PARAMS, 0)
    staged = staging.write_yolo_dataset(tmp_path / "final", train_tiles, None)
    cfg = yaml.safe_load(staged.yaml_path.read_text(encoding="utf-8"))
    assert cfg["val"] == cfg["train"]

import pytest
from tests.conftest import BUILD_A, BUILD_B, BUILD_C, DETECT_CONFIG

from layerwise.detect import data
from layerwise.detect.config import load_detect_config
from layerwise.errors import FoldOverlapError
from layerwise.ingest.pipeline import run_ingest


@pytest.fixture
def ingested(synthetic, ingest_cfg):
    for build in (BUILD_A, BUILD_B, BUILD_C):
        for layer in range(1, 4):
            synthetic.add_pb(layer, boxes=[("defects", 10, 10, 50, 40)], build=build)
        synthetic.add_pb(4, boxes=[], build=build)  # labeled, no defects
        synthetic.add_pb(5, boxes=None, build=build)  # unlabeled
    run_ingest(ingest_cfg)
    return load_detect_config(DETECT_CONFIG, "fixture")


def test_unlabeled_images_never_appear_in_any_set(ingested):
    plan = data.load_fold_plan(ingested)
    seen = {
        img.image_id
        for build in (BUILD_A, BUILD_B, BUILD_C)
        for img in data.labeled_images(ingested, [build])
    }
    assert not any("_00005_" in image_id for image_id in seen)
    assert len(plan.outer_folds) == 3


def test_labeled_no_defects_images_are_kept_with_zero_boxes(ingested):
    images = data.labeled_images(ingested, [BUILD_A])
    assert len(images) == 4
    assert sum(len(img.boxes) == 0 for img in images) == 1
    assert all(img.path.is_file() for img in images)


def test_per_build_counts_match_the_images_table(ingested):
    counts = data.labeled_counts(ingested)
    assert counts[BUILD_A] == {"images": 4, "boxes": 3}
    assert set(counts) == {BUILD_A, BUILD_B, BUILD_C}


def test_fold_plan_roles(ingested):
    plan = data.load_fold_plan(ingested)
    fold0 = plan.outer_folds[0]
    assert fold0.test_build == BUILD_A
    assert fold0.train_builds == [BUILD_B, BUILD_C]
    assert [(i.train_builds, i.validation_builds) for i in fold0.inner] == [
        ([BUILD_B], [BUILD_C]),
        ([BUILD_C], [BUILD_B]),
    ]


def test_build_with_two_roles_in_a_fold_stops_the_loader(ingested):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = ingested.ingest.processed_dir / "folds.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows.append({**rows[0], "role": "test" if rows[0]["role"] != "test" else "train"})
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(FoldOverlapError, match=rf"{rows[0]['build_id']}.*2 roles|2 roles"):
        data.load_fold_plan(ingested)

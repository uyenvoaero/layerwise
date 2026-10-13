from layerwise.ingest.folds import make_folds

A, B, C = "SI1A", "SI1B", "SI1C"


def rows_for(folds, level, outer, inner=None):
    return {
        r["build_id"]: r["role"]
        for r in folds
        if r["level"] == level and r["outer_fold"] == outer and r["inner_fold"] == inner
    }


def test_three_builds_give_21_rows():
    assert len(make_folds([A, B, C])) == 21  # 3 outer x (3 + 2 x 2)


def test_outer_fold_k_holds_out_build_k_sorted():
    folds = make_folds([C, A, B])  # unsorted input
    assert rows_for(folds, "outer", 0) == {A: "test", B: "train", C: "train"}
    assert rows_for(folds, "outer", 1) == {A: "train", B: "test", C: "train"}
    assert rows_for(folds, "outer", 2) == {A: "train", B: "train", C: "test"}


def test_inner_folds_train_on_one_build_validate_on_the_other():
    folds = make_folds([A, B, C])
    # outer fold 0 holds out A; the non-test builds are B, C
    assert rows_for(folds, "inner", 0, 0) == {B: "train", C: "validation"}
    assert rows_for(folds, "inner", 0, 1) == {B: "validation", C: "train"}
    assert A not in {r["build_id"] for r in folds if r["level"] == "inner" and r["outer_fold"] == 0}
    # outer fold 2 holds out C; the non-test builds are A, B
    assert rows_for(folds, "inner", 2, 0) == {A: "train", B: "validation"}
    assert rows_for(folds, "inner", 2, 1) == {A: "validation", B: "train"}


def test_outer_rows_have_null_inner_fold():
    folds = make_folds([A, B, C])
    assert all((r["inner_fold"] is None) == (r["level"] == "outer") for r in folds)


def test_two_builds_have_outer_folds_and_no_inner_folds():
    folds = make_folds([A, B])
    assert len(folds) == 4
    assert {r["level"] for r in folds} == {"outer"}
    assert rows_for(folds, "outer", 0) == {A: "test", B: "train"}


def test_one_build_has_no_folds():
    assert make_folds([A]) == []
    assert make_folds([]) == []


def test_regeneration_is_identical():
    assert make_folds([A, B, C]) == make_folds([C, B, A])


def test_every_build_has_exactly_one_role_per_fold():
    folds = make_folds([A, B, C])
    seen = {}
    for r in folds:
        key = (r["level"], r["outer_fold"], r["inner_fold"], r["build_id"])
        assert key not in seen
        seen[key] = r["role"]

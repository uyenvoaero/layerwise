from layerwise.ingest.checks import find_violations
from layerwise.ingest.folds import make_folds

A, B, C = "SI1A", "SI1B", "SI1C"


def test_valid_table_passes():
    assert find_violations(make_folds([A, B, C])) == []


def test_build_with_two_roles_in_an_outer_fold_fails():
    folds = make_folds([A, B, C])
    folds.append({"level": "outer", "outer_fold": 1, "inner_fold": None, "build_id": C,
                  "role": "test"})  # fmt: skip
    violations = find_violations(folds)
    assert len(violations) >= 1
    v = violations[0]
    assert (v.level, v.outer_fold, v.inner_fold, v.build_id) == ("outer", 1, None, C)
    assert C in v.message and "outer fold 1" in v.message


def test_build_with_two_roles_in_an_inner_fold_fails():
    folds = make_folds([A, B, C])
    folds.append({"level": "inner", "outer_fold": 0, "inner_fold": 1, "build_id": B,
                  "role": "train"})  # fmt: skip
    violations = find_violations(folds)
    assert any(v.build_id == B and v.inner_fold == 1 for v in violations)
    assert any("inner fold 1" in v.message and "outer fold 0" in v.message for v in violations)


def test_test_build_in_its_inner_folds_fails():
    folds = make_folds([A, B, C])
    folds.append({"level": "inner", "outer_fold": 0, "inner_fold": 0, "build_id": A,
                  "role": "validation"})  # fmt: skip
    violations = find_violations(folds)
    assert any(v.build_id == A and v.outer_fold == 0 and "test build" in v.message
               for v in violations)  # fmt: skip


def test_empty_table_passes():
    assert find_violations([]) == []

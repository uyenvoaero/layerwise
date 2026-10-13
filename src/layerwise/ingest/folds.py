from typing import Any


def make_folds(build_ids: list[str]) -> list[dict[str, Any]]:
    """Outer and inner leave-one-build-out folds over sorted build IDs; no randomness.

    Outer fold k holds out build k as `test`; the other builds are `train`. Inner fold j runs
    over the non-test builds only: it validates on the next non-test build (j + 1, wrapping)
    and trains on the rest, so with two non-test builds it trains on build j and validates on
    the other. With fewer than two non-test builds there are no inner folds; with one build
    there are no folds at all.
    """
    builds = sorted(set(build_ids))
    if len(builds) < 2:
        return []
    rows: list[dict[str, Any]] = []
    for k, test_build in enumerate(builds):
        for build in builds:
            rows.append(_row("outer", k, None, build, "test" if build == test_build else "train"))
        rest = [b for b in builds if b != test_build]
        if len(rest) < 2:
            continue
        for j in range(len(rest)):
            validation = rest[(j + 1) % len(rest)]
            for build in rest:
                rows.append(
                    _row("inner", k, j, build, "validation" if build == validation else "train")
                )
    rows.sort(
        key=lambda r: (
            r["level"],
            r["outer_fold"],
            -1 if r["inner_fold"] is None else r["inner_fold"],
            r["build_id"],
        )
    )
    return rows


def fold_notes(build_ids: list[str]) -> list[str]:
    """Report notes for subsets with too few builds for the full fold structure."""
    n = len(set(build_ids))
    if n < 2:
        return [f"{n} build(s) present: no folds were produced"]
    if n < 3:
        return [f"{n} builds present: outer folds only, no inner folds"]
    return []


def _row(level: str, outer: int, inner: int | None, build: str, role: str) -> dict[str, Any]:
    return {
        "level": level,
        "outer_fold": outer,
        "inner_fold": inner,
        "build_id": build,
        "role": role,
    }

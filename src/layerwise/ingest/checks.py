from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Violation:
    level: str
    outer_fold: int
    inner_fold: int | None
    build_id: str
    message: str


def _fold_name(level: str, outer: int, inner: int | None) -> str:
    if level == "outer":
        return f"outer fold {outer}"
    return f"inner fold {inner} of outer fold {outer}"


def find_violations(rows: list[dict[str, Any]]) -> list[Violation]:
    """Leakage checks on a fold table.

    1. Within one (level, outer_fold, inner_fold), a build must have exactly one role.
    2. The test build of an outer fold must not appear in any inner fold of that outer fold.
    """
    violations: list[Violation] = []
    roles: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        roles[(r["level"], r["outer_fold"], r["inner_fold"], r["build_id"])].append(r["role"])
    for (level, outer, inner, build), found in sorted(
        roles.items(),
        key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] is None else kv[0][2], kv[0][3]),
    ):
        if len(found) > 1:
            violations.append(
                Violation(
                    level, outer, inner, build,
                    f"build {build} has {len(found)} roles ({', '.join(sorted(set(found)))}) "
                    f"in {_fold_name(level, outer, inner)}",
                )
            )  # fmt: skip

    test_builds = {
        r["outer_fold"]: r["build_id"]
        for r in rows
        if r["level"] == "outer" and r["role"] == "test"
    }
    for r in rows:
        if r["level"] == "inner" and test_builds.get(r["outer_fold"]) == r["build_id"]:
            violations.append(
                Violation(
                    "inner", r["outer_fold"], r["inner_fold"], r["build_id"],
                    f"test build {r['build_id']} of outer fold {r['outer_fold']} appears in "
                    f"{_fold_name('inner', r['outer_fold'], r['inner_fold'])}",
                )
            )  # fmt: skip
    return violations

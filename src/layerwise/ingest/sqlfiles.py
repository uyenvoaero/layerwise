from pathlib import Path

import duckdb

SQL_DIR = Path(__file__).parent / "sql"


def load_sql(filename: str, section: str | None = None, directory: Path = SQL_DIR) -> str:
    """Read a SQL file; with `section`, return the part after the `-- name: <section>` marker."""
    text = (directory / filename).read_text(encoding="utf-8")
    if section is None:
        return text
    parts: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("-- name:"):
            current = line.removeprefix("-- name:").strip()
            parts[current] = []
        elif current is not None:
            parts[current].append(line)
    return "\n".join(parts[section])


def open_tables(directory: Path, names: list[str]) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with one view per Parquet table in `directory`."""
    con = duckdb.connect()
    for name in names:
        path = (directory / f"{name}.parquet").as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con

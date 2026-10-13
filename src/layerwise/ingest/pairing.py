from dataclasses import dataclass
from typing import Any

import duckdb
import pyarrow as pa

from layerwise.ingest import tables
from layerwise.ingest.sqlfiles import load_sql


@dataclass(frozen=True)
class PairResult:
    table: pa.Table
    duplicates: list[dict[str, Any]]

    @property
    def paired_layers(self) -> int:
        return sum(1 for status in self.table["pair_status"].to_pylist() if status == "paired")


def pair_layers(con: duckdb.DuckDBPyConnection) -> PairResult:
    """Pair PB and OT images by (build_id, layer_index); `con` must expose an `images` view.

    If several images share a key, the earliest by timestamp then filename is used and the
    others are returned as duplicates. Pairing is by layer only: no pixel alignment is implied.
    """
    pairs = con.execute(load_sql("layer_pairs.sql", "pairs")).to_arrow_table()
    dup_rows = con.execute(load_sql("layer_pairs.sql", "duplicates")).fetchall()
    duplicates = [
        {
            "build_id": build_id,
            "layer_index": int(layer_index),
            "modality": modality,
            "image_id": image_id,
            "kept_image_id": kept_image_id,
        }
        for build_id, layer_index, modality, image_id, kept_image_id in dup_rows
    ]
    return PairResult(pairs.select(tables.LAYER_PAIRS.names).cast(tables.LAYER_PAIRS), duplicates)

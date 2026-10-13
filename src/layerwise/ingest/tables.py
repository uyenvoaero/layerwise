from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _schema(*fields: tuple[str, pa.DataType, bool]) -> pa.Schema:
    return pa.schema([pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in fields])


IMAGES = _schema(
    ("image_id", pa.string(), False),
    ("dataset", pa.string(), False),
    ("build_id", pa.string(), False),
    ("modality", pa.string(), False),
    ("layer_index", pa.int32(), False),
    ("timestamp", pa.string(), True),
    ("z_mm", pa.float64(), True),
    ("path", pa.string(), False),
    ("width", pa.int32(), False),
    ("height", pa.int32(), False),
    ("mode", pa.string(), False),
    ("size_flag", pa.bool_(), False),
    ("label_status", pa.string(), False),
    ("use_for_detection", pa.bool_(), False),
)

BOXES = _schema(
    ("box_id", pa.string(), False),
    ("image_id", pa.string(), False),
    ("xmin", pa.int32(), False),
    ("ymin", pa.int32(), False),
    ("xmax", pa.int32(), False),
    ("ymax", pa.int32(), False),
    ("source_class", pa.string(), False),
    ("phase_category", pa.string(), False),
    ("fine_class", pa.string(), True),
    ("clipped", pa.bool_(), False),
)

CLASS_MAPPING = _schema(
    ("dataset", pa.string(), False),
    ("source_class", pa.string(), False),
    ("phase_category", pa.string(), False),
    ("fine_class", pa.string(), True),
    ("mapping_version", pa.string(), False),
)

LAYER_PAIRS = _schema(
    ("build_id", pa.string(), False),
    ("layer_index", pa.int32(), False),
    ("pb_image_id", pa.string(), True),
    ("ot_image_id", pa.string(), True),
    ("pair_status", pa.string(), False),
)

FOLDS = _schema(
    ("level", pa.string(), False),
    ("outer_fold", pa.int32(), False),
    ("inner_fold", pa.int32(), True),
    ("build_id", pa.string(), False),
    ("role", pa.string(), False),
)


def write_table(path: Path, rows: list[dict] | pa.Table, schema: pa.Schema) -> int:
    """Write a table with fixed options so reruns are byte-identical; returns the row count."""
    if isinstance(rows, pa.Table):
        table = rows.select(schema.names).cast(schema)
    else:
        table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=2**20)
    return table.num_rows

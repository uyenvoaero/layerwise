"""Draw the boxes of one ingested image on the image and save a PNG for a visual check.

Usage (DATA_ROOT must be set):
    uv run python scripts/show_sample.py --image-id <image_id> --out sample.png
    uv run python scripts/show_sample.py --build SI383820211201123521 --layer 1000 --modality OT
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb
from PIL import Image, ImageDraw

COLORS = {"recoating_phase_anomaly": (255, 64, 64), "melting_phase_anomaly": (255, 200, 0)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processed-dir", default="processed/aalto", help="Relative to DATA_ROOT.")
    parser.add_argument("--image-id")
    parser.add_argument("--build")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--modality", choices=["PB", "OT"], default="PB")
    parser.add_argument("--out", default="sample.png")
    args = parser.parse_args()

    data_root = Path(os.environ["DATA_ROOT"])
    processed = data_root / args.processed_dir
    con = duckdb.connect()
    images = (processed / "images.parquet").as_posix()
    boxes = (processed / "boxes.parquet").as_posix()

    if args.image_id:
        where, params = "image_id = ?", [args.image_id]
    elif args.build and args.layer is not None:
        where = "build_id = ? AND layer_index = ? AND modality = ?"
        params = [args.build, args.layer, args.modality]
    else:
        parser.error("give --image-id, or --build and --layer")

    row = con.execute(
        f"SELECT image_id, path, label_status FROM read_parquet('{images}') WHERE {where} "
        "ORDER BY image_id LIMIT 1",
        params,
    ).fetchone()
    if row is None:
        print("no matching image", file=sys.stderr)
        return 1
    image_id, rel_path, status = row
    rows = con.execute(
        f"SELECT xmin, ymin, xmax, ymax, source_class, phase_category "
        f"FROM read_parquet('{boxes}') WHERE image_id = ? ORDER BY box_id",
        [image_id],
    ).fetchall()

    image = Image.open(data_root / rel_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for xmin, ymin, xmax, ymax, source_class, phase in rows:
        color = COLORS.get(phase, (0, 255, 0))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        draw.text((xmin + 3, max(ymin - 12, 0)), source_class, fill=color)
    image.save(args.out)
    print(f"{image_id}: {status}, {len(rows)} boxes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

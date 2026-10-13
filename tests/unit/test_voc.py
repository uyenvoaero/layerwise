import pytest

from layerwise.ingest.adapters.base import RawBox
from layerwise.ingest.voc import clip_boxes, read_voc


def _write(path, boxes, path_text="C:\\Users\\author\\x.jpg"):
    objects = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, a, b, c, d in boxes
    )
    path.write_text(
        f"<annotation><folder>x</folder><path>{path_text}</path>"
        f"<size><width>100</width><height>50</height></size>{objects}</annotation>",
        encoding="utf-8",
    )


def test_read_voc_ignores_path_and_returns_boxes(tmp_path):
    xml = tmp_path / "a.xml"
    _write(xml, [("defects", 1, 2, 30, 40), ("overheated", 5.0, 6.0, 7.0, 8.0)])
    boxes = read_voc(xml)
    assert boxes == [
        RawBox(1, 2, 30, 40, "defects"),
        RawBox(5, 6, 7, 8, "overheated"),
    ]


def test_read_voc_zero_objects(tmp_path):
    xml = tmp_path / "a.xml"
    _write(xml, [])
    assert read_voc(xml) == []


def test_read_voc_malformed_raises(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<annotation><oops", encoding="utf-8")
    with pytest.raises(ValueError):
        read_voc(xml)


def test_clip_inside_box_unchanged():
    result = clip_boxes([RawBox(1, 2, 30, 40, "defects")], width=100, height=50)
    (box,) = result.boxes
    assert (box.xmin, box.ymin, box.xmax, box.ymax, box.clipped) == (1, 2, 30, 40, False)
    assert (result.ingested, result.clipped, result.dropped) == (1, 0, 0)


def test_clip_out_of_image_box():
    result = clip_boxes([RawBox(-5, 10, 120, 60, "defects")], width=100, height=50)
    (box,) = result.boxes
    assert (box.xmin, box.ymin, box.xmax, box.ymax, box.clipped) == (0, 10, 100, 50, True)
    assert (result.ingested, result.clipped, result.dropped) == (1, 1, 0)


def test_zero_area_boxes_are_dropped_and_counted():
    raw = [
        RawBox(10, 10, 10, 20, "defects"),  # zero width
        RawBox(200, 10, 300, 20, "defects"),  # fully outside: zero area after clipping
        RawBox(1, 1, 5, 5, "defects"),
    ]
    result = clip_boxes(raw, width=100, height=50)
    assert [b.index for b in result.boxes] == [2]
    assert (result.ingested, result.dropped) == (1, 2)


def test_invariant_after_clipping():
    raw = [RawBox(-10, -10, 500, 500, "defects"), RawBox(3, 4, 99, 49, "overheated")]
    for box in clip_boxes(raw, width=100, height=50).boxes:
        assert 0 <= box.xmin < box.xmax <= 100
        assert 0 <= box.ymin < box.ymax <= 50

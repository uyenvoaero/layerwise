import pytest
from tests.conftest import BUILD_A, BUILD_B, UNKNOWN_BUILD

from layerwise.ingest.adapters.aalto import AaltoAdapter


def _discover(cfg):
    adapter = AaltoAdapter(cfg)
    return adapter, list(adapter.discover())


def test_pb_timestamp_with_and_without_fraction(ingest_cfg, synthetic):
    synthetic.add_pb(218, timestamp="20240318T140107")
    synthetic.add_pb(219, timestamp="20240318T140109.125")
    _, records = _discover(ingest_cfg)
    by_layer = {r.layer_index: r for r in records}
    assert by_layer[218].timestamp == "20240318T140107"
    assert by_layer[219].timestamp == "20240318T140109.125"
    assert all(r.modality == "PB" and r.z_mm is None for r in records)
    assert by_layer[218].image_id == f"{BUILD_A}_00218_20240318T140107"


def test_ot_layer_index_and_height(ingest_cfg, synthetic):
    synthetic.add_ot(1000, z="039_400")
    _, records = _discover(ingest_cfg)
    (record,) = records
    assert record.modality == "OT"
    assert record.layer_index == 1000
    assert record.z_mm == pytest.approx(39.4)
    assert record.timestamp is None
    assert record.image_id == f"{BUILD_A}_1000_039_400_Int_32F"


def test_unparseable_names_are_reported_not_ingested(ingest_cfg, synthetic):
    synthetic.add_pb(1)
    bad = synthetic.add_bad_name("PB", "garbage")
    bad_ot = synthetic.add_bad_name("OT", "also_garbage")
    adapter, records = _discover(ingest_cfg)
    assert len(records) == 1
    assert sorted(p.name for p in adapter.unparsed) == sorted([bad, bad_ot])


def test_unknown_build_is_reported_and_skipped(ingest_cfg, synthetic):
    synthetic.add_pb(1, build=BUILD_A)
    synthetic.add_pb(2, build=UNKNOWN_BUILD)
    synthetic.add_ot(3, build=UNKNOWN_BUILD)
    synthetic.add_pb(4, build=BUILD_B)
    adapter, records = _discover(ingest_cfg)
    assert {r.build_id for r in records} == {BUILD_A, BUILD_B}
    assert len(adapter.unknown_build) == 2
    assert all(UNKNOWN_BUILD in p.name for p in adapter.unknown_build)


def test_labels_none_without_xml_and_empty_with_empty_xml(ingest_cfg, synthetic):
    synthetic.add_pb(1)
    synthetic.add_pb(2, boxes=[])
    synthetic.add_pb(3, boxes=[("defects", 1, 2, 10, 20)])
    adapter, records = _discover(ingest_cfg)
    labels = {r.layer_index: adapter.labels(r) for r in records}
    assert labels[1] is None
    assert labels[2] == []
    (box,) = labels[3]
    assert (box.xmin, box.ymin, box.xmax, box.ymax, box.source_class) == (1, 2, 10, 20, "defects")


def test_orphan_xml(ingest_cfg, synthetic):
    synthetic.add_pb(1, boxes=[("defects", 1, 1, 5, 5)])
    synthetic.add_orphan_xml("PB", synthetic.pb_stem(77))
    adapter, _ = _discover(ingest_cfg)
    assert [p.stem for p in adapter.orphan_xml()] == [synthetic.pb_stem(77)]


def test_source_files_and_info(ingest_cfg, synthetic):
    synthetic.add_pb(1, boxes=[])
    synthetic.add_ot(2)
    adapter, _ = _discover(ingest_cfg)
    images, xmls = adapter.source_files()
    assert len(images) == 2 and len(xmls) == 1
    info = adapter.info()
    assert info["license"] == "CC-BY-4.0"
    assert info["doi"] == "10.5281/zenodo.14996805"
    assert adapter.source_classes() == ["defects", "overheated"]


def test_missing_source_dir_is_config_error(ingest_cfg):
    from layerwise.errors import ConfigError

    with pytest.raises(ConfigError, match="source"):
        list(AaltoAdapter(ingest_cfg).discover())

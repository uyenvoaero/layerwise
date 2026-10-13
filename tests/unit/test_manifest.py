import hashlib
import json

from layerwise.manifest import (
    build_manifest,
    config_hash,
    dumps_json,
    input_hash,
    write_json,
)


def _make_files(root):
    img = root / "PB" / "a.jpg"
    xml = root / "PB_label" / "a.xml"
    img.parent.mkdir()
    xml.parent.mkdir()
    img.write_bytes(b"12345")
    xml.write_text("<annotation/>", encoding="utf-8")
    return img, xml


def test_input_hash_is_stable(tmp_path):
    img, xml = _make_files(tmp_path)
    assert input_hash(tmp_path, [img], [xml]) == input_hash(tmp_path, [img], [xml])


def test_input_hash_uses_size_for_images_and_content_for_xml(tmp_path):
    img, xml = _make_files(tmp_path)
    before = input_hash(tmp_path, [img], [xml])
    img.write_bytes(b"12345")  # same size, same hash
    assert input_hash(tmp_path, [img], [xml]) == before
    img.write_bytes(b"123456")  # size change
    assert input_hash(tmp_path, [img], [xml]) != before
    img.write_bytes(b"12345")
    xml.write_text("<annotation>x</annotation>", encoding="utf-8")  # content change
    assert input_hash(tmp_path, [img], [xml]) != before


def test_input_hash_ignores_input_order(tmp_path):
    img, xml = _make_files(tmp_path)
    img2 = tmp_path / "PB" / "b.jpg"
    img2.write_bytes(b"1")
    assert input_hash(tmp_path, [img, img2], [xml]) == input_hash(tmp_path, [img2, img], [xml])


def test_input_hash_uses_relative_paths(tmp_path):
    img, xml = _make_files(tmp_path)
    other = tmp_path.parent / (tmp_path.name + "_copy")
    other.mkdir()
    img2, xml2 = _make_files(other)
    assert input_hash(tmp_path, [img], [xml]) == input_hash(other, [img2], [xml2])
    assert hashlib.sha256(b"x").hexdigest() not in input_hash(tmp_path, [img], [xml])


def test_json_format(tmp_path):
    text = dumps_json({"b": 1, "a": {"d": 2, "c": 3}})
    assert text == '{\n  "a": {\n    "c": 3,\n    "d": 2\n  },\n  "b": 1\n}\n'
    path = tmp_path / "x.json"
    write_json(path, {"b": 1, "a": 2})
    assert path.read_bytes().endswith(b"}\n")
    assert b"\r" not in path.read_bytes()


def test_manifest_keys_and_no_timestamps():
    cfg = {"source_dir": "aalto", "seed": None}
    manifest = build_manifest(
        dataset="aalto",
        source={"name": "n", "doi": "d", "license": "CC-BY-4.0"},
        adapter_version="1",
        input_hash_value="h",
        config=cfg,
        mapping_version="1",
        seed=None,
        outputs={"images": 1},
        metrics={"images": 1},
    )
    assert set(manifest) == {
        "dataset",
        "source",
        "code_version",
        "adapter_version",
        "input_hash",
        "config_hash",
        "config",
        "mapping_version",
        "seed",
        "outputs",
        "metrics",
    }
    assert set(manifest["source"]) == {"name", "doi", "license"}
    assert manifest["seed"] is None
    assert manifest["config_hash"] == config_hash(cfg)
    text = json.dumps(manifest)
    assert "timestamp" not in text and "created" not in text


def test_config_hash_changes_with_config():
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

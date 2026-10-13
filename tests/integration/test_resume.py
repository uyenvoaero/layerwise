import json

import pytest
import yaml
from tests.conftest import DETECT_CONFIG

pytest.importorskip("ultralytics")

import ultralytics  # noqa: E402

from layerwise.detect import protocol  # noqa: E402
from layerwise.detect.config import load_detect_config  # noqa: E402
from layerwise.detect.models import make_detector  # noqa: E402
from layerwise.detect.models.yolo import YoloDetector  # noqa: E402
from layerwise.ingest.pipeline import run_ingest  # noqa: E402

pytestmark = pytest.mark.slow


class Interrupted(Exception):
    pass


@pytest.fixture
def cfg(fixture_data_root, ingest_cfg, tmp_path):
    run_ingest(ingest_cfg)
    raw = yaml.safe_load(DETECT_CONFIG.read_text(encoding="utf-8"))
    raw["ingest_config"] = str(DETECT_CONFIG.parents[1] / "ingest" / "base.yaml")
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    fixture = yaml.safe_load((DETECT_CONFIG.parent / "fixture.yaml").read_text(encoding="utf-8"))
    fixture["train"] = {"max_epochs": 2, "patience": 5}
    (tmp_path / "fixture.yaml").write_text(yaml.safe_dump(fixture), encoding="utf-8")
    return load_detect_config(tmp_path / "base.yaml", "fixture")


def factory(cfg):
    return lambda model, candidate: make_detector(cfg, model, candidate)


def test_rerun_after_interruption_skips_the_finished_inner_job(cfg, monkeypatch):
    original = YoloDetector.fit
    calls = []

    def flaky(self, *args, **kwargs):
        if len(calls) == 1:
            raise Interrupted
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(YoloDetector, "fit", flaky)
    with pytest.raises(Interrupted):
        protocol.run_protocol(cfg, "yolo", factory(cfg), folds=[0])
    monkeypatch.setattr(YoloDetector, "fit", original)
    summary = protocol.run_protocol(cfg, "yolo", factory(cfg), folds=[0])
    events = summary.outcomes[0].events
    assert [e["status"] for e in events if e["stage"] == "inner"] == ["skipped", "ran"]
    assert summary.outcomes[0].heldout_result is not None


def test_interrupted_training_resumes_from_the_last_checkpoint(cfg, monkeypatch):
    original = ultralytics.YOLO

    class Interrupting(original):
        def train(self, *args, **kwargs):
            if not kwargs.get("resume"):

                def stop(trainer):
                    if trainer.epoch == 0:  # checkpoint of epoch 1 is on disk
                        raise Interrupted

                self.add_callback("on_model_save", stop)
            return super().train(*args, **kwargs)

    monkeypatch.setattr(ultralytics, "YOLO", Interrupting)
    with pytest.raises(Interrupted):
        protocol.run_protocol(cfg, "yolo", factory(cfg), folds=[0])
    job = cfg.runs_dir / "yolo/outer_0/inner_0/yolov8n-scratch-t256"
    assert (job / "weights/last.pt").exists() and not (job / "result.json").exists()

    monkeypatch.setattr(ultralytics, "YOLO", original)
    protocol.run_protocol(cfg, "yolo", factory(cfg), folds=[0])
    fit = json.loads((job / "fit.json").read_text(encoding="utf-8"))
    assert fit["epochs_run"] == 2  # epoch 1 kept, only epoch 2 trained after the restart
    assert fit["history"][0]["epoch"] == 1.0

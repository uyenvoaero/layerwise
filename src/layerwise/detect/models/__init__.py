from layerwise.detect.config import Candidate, DetectConfig
from layerwise.detect.models.base import Detector


def make_detector(cfg: DetectConfig, model: str, candidate: Candidate) -> Detector:
    """Build the detector for a model name; heavy libraries are imported only when needed."""
    if model == "yolo":
        from layerwise.detect.models.yolo import YoloDetector

        return YoloDetector(cfg, candidate)
    if model == "frcnn":
        from layerwise.detect.models.frcnn import FasterRcnnDetector

        return FasterRcnnDetector(cfg, candidate)
    if model == "contrast":
        from layerwise.detect.models.contrast import ContrastDetector

        return ContrastDetector(cfg, candidate)
    raise ValueError(f"unknown model {model!r}")

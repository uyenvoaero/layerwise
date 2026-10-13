from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """One source image as found on disk; width/height are read later by the pipeline."""

    image_id: str
    build_id: str
    modality: str  # "PB" or "OT"
    layer_index: int
    path: Path
    timestamp: str | None = None  # PB only
    z_mm: float | None = None  # OT only


@dataclass(frozen=True)
class RawBox:
    """A bounding box as written in the source label file (not yet clipped)."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source_class: str


class DatasetAdapter(ABC):
    """Interface between a dataset's on-disk layout and the uniform ingest tables.

    The Aalto adapter implements it now; the Peregrine adapter (pixel masks, part regions,
    scan parameters, sensor readings, tensile results) implements it when that data is on disk.
    """

    name: str
    version: str

    @abstractmethod
    def discover(self) -> Iterator[ImageRecord]:
        """Yield every image with build, modality, layer index and path."""

    @abstractmethod
    def labels(self, image: ImageRecord) -> list[RawBox] | None:
        """Return the boxes of an image, or None when the image has no label file."""

    @abstractmethod
    def source_classes(self) -> list[str]:
        """Source class names this adapter can emit."""

    @abstractmethod
    def info(self) -> dict[str, str]:
        """Dataset name, DOI and license for the manifest."""

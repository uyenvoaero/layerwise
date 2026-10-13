import re
from collections.abc import Iterator
from pathlib import Path

from layerwise.config import IngestConfig
from layerwise.errors import ConfigError
from layerwise.ingest.adapters.base import DatasetAdapter, ImageRecord, RawBox
from layerwise.ingest.voc import read_voc

# PB: <build>_<layer, 5 digits>_<YYYYMMDD>T<HHMMSS>[.<fraction>].jpg
#   e.g. SI383820240318120348_00218_20240318T140107.jpg (2 files have no fractional seconds)
# OT: <build>_<layer>_<z int>_<z fraction>_Int_32F.jpg
#   e.g. SI383820211201123521_1000_039_400_Int_32F.jpg -> layer 1000, z = 39.400 mm
# For OT the first numeric field is the layer index; the next two fields form the height.
PB_PATTERN = re.compile(
    r"^(?P<build>SI\d+)_(?P<layer>\d{5})_(?P<timestamp>\d{8}T\d{6}(?:\.\d+)?)\.jpg$",
    re.IGNORECASE,
)
OT_PATTERN = re.compile(
    r"^(?P<build>SI\d+)_(?P<layer>\d+)_(?P<z_int>\d+)_(?P<z_frac>\d+)_Int_32F\.jpg$",
    re.IGNORECASE,
)

MODALITIES = ("PB", "OT")


class AaltoAdapter(DatasetAdapter):
    """Aalto L-PBF powder bed (PB) and optical tomography (OT) images with Pascal VOC boxes."""

    name = "aalto"
    version = "1"
    SOURCE_CLASSES = ("defects", "overheated")

    def __init__(self, cfg: IngestConfig):
        self._cfg = cfg
        self._scanned = False
        self._records: list[ImageRecord] = []
        self._image_files: list[Path] = []
        self._xml_files: list[Path] = []
        self.unparsed: list[Path] = []
        self.unknown_build: list[Path] = []

    def _scan(self) -> None:
        if self._scanned:
            return
        source = self._cfg.source_dir
        if not source.is_dir():
            raise ConfigError(f"source folder not found: {source}")
        known = set(self._cfg.known_builds)
        for modality in MODALITIES:
            pattern = PB_PATTERN if modality == "PB" else OT_PATTERN
            for path in sorted((source / modality).glob("*.jpg")):
                self._image_files.append(path)
                match = pattern.match(path.name)
                if match is None:
                    self.unparsed.append(path)
                elif match["build"] not in known:
                    self.unknown_build.append(path)
                elif modality == "PB":
                    self._records.append(
                        ImageRecord(
                            image_id=path.stem,
                            build_id=match["build"],
                            modality="PB",
                            layer_index=int(match["layer"]),
                            path=path,
                            timestamp=match["timestamp"],
                        )
                    )
                else:
                    self._records.append(
                        ImageRecord(
                            image_id=path.stem,
                            build_id=match["build"],
                            modality="OT",
                            layer_index=int(match["layer"]),
                            path=path,
                            z_mm=float(f"{match['z_int']}.{match['z_frac']}"),
                        )
                    )
            self._xml_files.extend(sorted((source / f"{modality}_label").glob("*.xml")))
        self._scanned = True

    def discover(self) -> Iterator[ImageRecord]:
        self._scan()
        yield from self._records

    def source_files(self) -> tuple[list[Path], list[Path]]:
        """All image and XML files found, including unparsed and unknown-build ones."""
        self._scan()
        return list(self._image_files), list(self._xml_files)

    def label_path(self, image: ImageRecord) -> Path:
        return self._cfg.source_dir / f"{image.modality}_label" / f"{image.path.stem}.xml"

    def labels(self, image: ImageRecord) -> list[RawBox] | None:
        path = self.label_path(image)
        if not path.exists():
            return None
        return read_voc(path)

    def orphan_xml(self) -> list[Path]:
        """XML files whose image does not exist."""
        self._scan()
        image_stems = {(p.parent.name, p.stem) for p in self._image_files}
        return [
            p
            for p in self._xml_files
            if (p.parent.name.removesuffix("_label"), p.stem) not in image_stems
        ]

    def source_classes(self) -> list[str]:
        return list(self.SOURCE_CLASSES)

    def info(self) -> dict[str, str]:
        return {
            "name": "Aalto L-PBF powder bed and optical tomography images",
            "doi": "10.5281/zenodo.14996805",
            "license": "CC-BY-4.0",
        }

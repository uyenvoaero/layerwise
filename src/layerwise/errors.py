class ConfigError(Exception):
    """Configuration problem (missing DATA_ROOT, bad YAML, missing source folder). Exit code 2."""


class UnmappedClassError(Exception):
    """A source class has no entry in the class mapping. Exit code 3."""


class InputErrorsExceededError(Exception):
    """Fraction of unreadable or corrupt input files is above failure_threshold. Exit code 4."""


class FoldOverlapError(Exception):
    """The fold table gives a build two roles in one fold (leakage). Exit code 1."""


class UnreadableImagesError(Exception):
    """Fraction of unreadable labeled images is above failure_threshold. Exit code 3."""

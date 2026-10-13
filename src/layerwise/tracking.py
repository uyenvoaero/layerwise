import os
from contextlib import contextmanager
from typing import Any

from layerwise.config import IngestConfig


def log_run(cfg: IngestConfig, manifest: dict[str, Any]) -> None:
    """Log a manifest to the local MLflow file store; no-op when tracking is off."""
    if not cfg.mlflow_enabled:
        return
    # Newer MLflow versions refuse the local file store unless it is explicitly allowed.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow_dir.resolve().as_uri())
    mlflow.set_experiment(f"ingest-{manifest['dataset']}")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "input_hash": manifest["input_hash"],
                "config_hash": manifest["config_hash"],
                "mapping_version": manifest["mapping_version"],
                "code_version": manifest["code_version"],
                "adapter_version": manifest["adapter_version"],
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in manifest["metrics"].items()})


class _NoRun:
    run_id = None

    def log(self, params: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None):
        pass


class _MlflowRun:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def log(self, params: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None):
        import mlflow

        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        numeric = {k: float(v) for k, v in (metrics or {}).items() if v is not None}
        if numeric:
            mlflow.log_metrics(numeric)


@contextmanager
def start_run(cfg: Any, experiment: str, run_name: str, nested: bool = False):
    """MLflow run for any config with `mlflow_enabled` and `mlflow_dir`; no-op when tracking is off.

    Yields an object with `run_id` (None when off) and `log(params, metrics)`.
    """
    if not cfg.mlflow_enabled:
        yield _NoRun()
        return
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow_dir.resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield _MlflowRun(run.info.run_id)

.PHONY: ingest check-splits make-fixture test lint train baseline predict benchmark report smoke sync-up sync-down

ingest:
	uv run layerwise ingest

check-splits:
	uv run layerwise check-splits

make-fixture:
	uv run layerwise make-fixture

test:
	uv run pytest

lint:
	uv run ruff check .

train:
	uv run layerwise train

baseline:
	uv run layerwise baseline

predict:
	uv run layerwise predict

benchmark:
	uv run layerwise benchmark

report:
	uv run layerwise report

smoke:
	uv run python scripts/smoke.py

# S3_URI comes from the environment; the library has no AWS code.
sync-up:
	aws s3 sync "$(DATA_ROOT)/runs" "$(S3_URI)/runs"
	aws s3 sync "$(DATA_ROOT)/predictions" "$(S3_URI)/predictions"
	aws s3 sync "$(DATA_ROOT)/reports" "$(S3_URI)/reports"

sync-down:
	aws s3 sync "$(S3_URI)/runs" "$(DATA_ROOT)/runs"
	aws s3 sync "$(S3_URI)/predictions" "$(DATA_ROOT)/predictions"
	aws s3 sync "$(S3_URI)/reports" "$(DATA_ROOT)/reports"

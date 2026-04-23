"""Promote the registered model and prepare local serving artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from eyewear_pipeline.config import ProjectConfig, ensure_dirs  # noqa: E402
from eyewear_pipeline.mlops import ModelMetadata, write_model_metadata  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a registered model and prepare local serving artifacts.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15040"))
    parser.add_argument("--run-manifest", type=Path, default=ROOT / "artifacts" / "train_smoke_run.json")
    parser.add_argument("--registration-manifest", type=Path, default=ROOT / "artifacts" / "registered_model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "promotion.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    ensure_dirs(cfg)

    run_payload = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    registration_payload = json.loads(args.registration_manifest.read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registration_payload["registered_model"],
        version=registration_payload["version"],
        stage="Production",
        archive_existing_versions=True,
    )

    downloaded_model = mlflow.artifacts.download_artifacts(artifact_uri=run_payload["raw_model_uri"])
    downloaded_threshold = mlflow.artifacts.download_artifacts(artifact_uri=run_payload["threshold_uri"])
    shutil.copy2(downloaded_model, cfg.baseline_model_path)
    shutil.copy2(downloaded_threshold, cfg.threshold_file_path)

    metadata = ModelMetadata(
        model_type="baseline",
        model_path=str(cfg.baseline_model_path),
        confidence_threshold=float(run_payload["threshold"]),
        threshold_version="f1",
        model_version=f"v{registration_payload['version']}",
        registry_stage="production",
        dataset_version=run_payload["dataset_version"],
        run_id=run_payload["run_id"],
        registered_model=registration_payload["registered_model"],
        source_uri=run_payload["raw_model_uri"],
    )
    write_model_metadata(cfg.model_metadata_path, metadata)

    payload = {
        "model_path": str(cfg.baseline_model_path),
        "threshold_path": str(cfg.threshold_file_path),
        "metadata_path": str(cfg.model_metadata_path),
        "registered_model": registration_payload["registered_model"],
        "version": registration_payload["version"],
        "run_id": run_payload["run_id"],
        "registry_stage": "Production",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

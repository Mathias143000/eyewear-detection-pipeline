"""Run the baseline smoke training loop and log it into MLflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from eyewear_pipeline.baseline import train_baseline  # noqa: E402
from eyewear_pipeline.calibration import save_threshold  # noqa: E402
from eyewear_pipeline.config import ProjectConfig, ensure_dirs  # noqa: E402
from eyewear_pipeline.mlops import calibrate_threshold_from_csv, evaluate_model  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic train/eval smoke and track it in MLflow.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    parser.add_argument("--experiment-name", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "eyewear-demo"))
    parser.add_argument("--dataset-version", default="synthetic-smoke-v1")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=48)
    parser.add_argument("--output-manifest", type=Path, default=ROOT / "artifacts" / "train_smoke_run.json")
    return parser.parse_args()


def maybe_prepare_data(samples_per_class: int) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_demo_data.py"),
        "--samples-per-class",
        str(samples_per_class),
        "--force",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    ensure_dirs(cfg)

    if args.prepare_data:
        maybe_prepare_data(args.samples_per_class)

    artifact = train_baseline(ROOT / "data" / "splits" / "train.csv")
    artifact.save(cfg.baseline_model_path)

    threshold, threshold_f1 = calibrate_threshold_from_csv(
        val_csv=ROOT / "data" / "splits" / "val.csv",
        model_path=cfg.baseline_model_path,
        model_type="baseline",
    )
    save_threshold(cfg.threshold_file_path, threshold, metric_name="f1")

    evaluation = evaluate_model(
        test_csv=ROOT / "data" / "splits" / "test.csv",
        model_path=cfg.baseline_model_path,
        model_type="baseline",
        threshold=threshold,
        confusion_matrix_path=cfg.reports_dir / "confusion_matrix.png",
    )
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="baseline-smoke") as run:
        mlflow.log_params(
            {
                "model_type": "baseline",
                "dataset_version": args.dataset_version,
                "train_csv": "data/splits/train.csv",
                "val_csv": "data/splits/val.csv",
                "test_csv": "data/splits/test.csv",
                "threshold_version": "f1",
            }
        )
        mlflow.log_metrics(
            {
                "precision": float(evaluation["precision"]),
                "recall": float(evaluation["recall"]),
                "f1": float(evaluation["f1"]),
                "roc_auc": float(evaluation["roc_auc"]),
                "threshold": float(threshold),
                "threshold_f1": float(threshold_f1),
            }
        )
        mlflow.log_artifact(str(cfg.baseline_model_path), artifact_path="serving-model")
        mlflow.log_artifact(str(cfg.threshold_file_path), artifact_path="serving-threshold")
        mlflow.log_artifact(str(cfg.reports_dir / "metrics.json"), artifact_path="reports")
        mlflow.log_artifact(str(cfg.reports_dir / "confusion_matrix.png"), artifact_path="reports")
        dataset_manifest = ROOT / "artifacts" / "dataset_manifest.json"
        if dataset_manifest.exists():
            mlflow.log_artifact(str(dataset_manifest), artifact_path="dataset")
        mlflow.sklearn.log_model(artifact.pipeline, artifact_path="model")

        payload = {
            "tracking_uri": args.tracking_uri,
            "experiment_name": args.experiment_name,
            "run_id": run.info.run_id,
            "dataset_version": args.dataset_version,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "raw_model_uri": f"runs:/{run.info.run_id}/serving-model/{cfg.baseline_model_path.name}",
            "threshold_uri": f"runs:/{run.info.run_id}/serving-threshold/{cfg.threshold_file_path.name}",
            "threshold": threshold,
            "metrics": evaluation,
        }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

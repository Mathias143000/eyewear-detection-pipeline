from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the latest smoke model in MLflow Model Registry.")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15040"))
    parser.add_argument("--registered-model", default=os.getenv("MLFLOW_REGISTERED_MODEL", "eyewear-classifier"))
    parser.add_argument("--run-manifest", type=Path, default=ROOT / "artifacts" / "train_smoke_run.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "registered_model.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.run_manifest.read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    try:
        client.create_registered_model(args.registered_model)
    except RestException:
        pass

    model_version = mlflow.register_model(model_uri=payload["model_uri"], name=args.registered_model)
    for _ in range(30):
        current = client.get_model_version(args.registered_model, model_version.version)
        if getattr(current, "status", "READY") == "READY":
            break
        time.sleep(1)

    current = client.get_model_version(args.registered_model, model_version.version)
    response = {
        "registered_model": args.registered_model,
        "version": current.version,
        "run_id": current.run_id,
        "source": current.source,
        "current_stage": current.current_stage,
        "tracking_uri": args.tracking_uri,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(response, indent=2), encoding="utf-8")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eyewear_pipeline.calibration import best_threshold_by_f1, save_threshold
from eyewear_pipeline.config import ProjectConfig
from eyewear_pipeline.inference import EyewearPredictor
from eyewear_pipeline.metrics import classification_metrics
from eyewear_pipeline.baseline import train_baseline
from eyewear_pipeline.torch_model import train_torch_model


@dataclass(slots=True)
class Experiment:
    name: str
    model_type: str
    model_name: str | None = None
    model_path: Path | None = None
    epochs: int = 2
    batch_size: int = 16
    lr: float = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible experiment set")
    parser.add_argument("--train-csv", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/experiment_table.csv"))
    return parser.parse_args()


def evaluate_split(csv_path: Path, predictor: EyewearPredictor) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    y_true, y_pred, y_score = [], [], []
    for row in df.itertuples(index=False):
        image = cv2.imread(str(row.image_path))
        if image is None:
            continue
        preds = predictor.predict_image(image)
        if not preds:
            continue
        top = max(preds, key=lambda x: x.confidence)
        y_true.append(int(row.label))
        y_pred.append(int(top.label_id))
        y_score.append(float(top.score_positive))
    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}
    return classification_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    threshold_file = Path("artifacts/threshold.json")

    experiments = [
        Experiment(name="baseline_lr", model_type="baseline", model_path=cfg.models_dir / "baseline_glasses.joblib"),
        Experiment(
            name="torch_mnv3_small_e2",
            model_type="torch",
            model_name="mobilenet_v3_small",
            model_path=cfg.models_dir / "torch_mnv3_small_e2.pt",
            epochs=2,
            batch_size=16,
            lr=1e-3,
        ),
        Experiment(
            name="torch_mnv3_small_e4",
            model_type="torch",
            model_name="mobilenet_v3_small",
            model_path=cfg.models_dir / "torch_mnv3_small_e4.pt",
            epochs=4,
            batch_size=16,
            lr=7e-4,
        ),
        Experiment(
            name="torch_efficientnet_b0_e8",
            model_type="torch",
            model_name="efficientnet_b0",
            model_path=cfg.models_dir / "torch_efficientnet_b0_e8.pt",
            epochs=8,
            batch_size=24,
            lr=3e-4,
        ),
    ]

    rows: list[dict[str, str | float]] = []
    for exp in experiments:
        row: dict[str, str | float] = {
            "experiment": exp.name,
            "model_type": exp.model_type,
            "model_name": exp.model_name or "baseline_lr",
            "epochs": exp.epochs if exp.model_type == "torch" else 0,
            "batch_size": exp.batch_size if exp.model_type == "torch" else 0,
            "lr": exp.lr if exp.model_type == "torch" else 0.0,
            "status": "ok",
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "model_path": "",
            "note": "",
        }
        try:
            if exp.model_type == "baseline":
                artifact = train_baseline(args.train_csv)
                model_path = exp.model_path or cfg.baseline_model_path
                artifact.save(model_path)
            else:
                model_path = exp.model_path or cfg.torch_model_path
                result = train_torch_model(
                    train_csv=args.train_csv,
                    val_csv=args.val_csv,
                    model_path=model_path,
                    model_name=exp.model_name or "mobilenet_v3_small",
                    epochs=exp.epochs,
                    batch_size=exp.batch_size,
                    lr=exp.lr,
                )
                row["note"] = json.dumps(result)

            predictor_val = EyewearPredictor(model_path=model_path, model_type=exp.model_type, confidence_threshold=0.5)
            val_df = pd.read_csv(args.val_csv)
            y_true_val, y_score_val = [], []
            for rec in val_df.itertuples(index=False):
                im = cv2.imread(str(rec.image_path))
                if im is None:
                    continue
                preds = predictor_val.predict_image(im)
                if not preds:
                    continue
                top = max(preds, key=lambda x: x.confidence)
                y_true_val.append(int(rec.label))
                y_score_val.append(float(top.score_positive))

            thr, _ = best_threshold_by_f1(np.array(y_true_val), np.array(y_score_val))
            save_threshold(threshold_file, thr, metric_name="f1")
            predictor_test = EyewearPredictor(
                model_path=model_path,
                model_type=exp.model_type,
                confidence_threshold=thr,
            )
            metrics = evaluate_split(args.test_csv, predictor_test)
            row.update(metrics)
            row["threshold"] = thr
            row["model_path"] = str(model_path)
        except Exception as exc:
            row["status"] = "failed"
            row["note"] = str(exc)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

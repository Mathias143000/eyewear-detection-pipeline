from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eyewear_pipeline.calibration import best_threshold_by_f1, save_threshold
from eyewear_pipeline.config import ProjectConfig
from eyewear_pipeline.inference import EyewearPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate classification threshold on validation split")
    parser.add_argument("--val-csv", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--model-type", choices=["baseline", "torch"], default="baseline")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/threshold.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    model_path = args.model_path or (
        cfg.baseline_model_path if args.model_type == "baseline" else cfg.torch_model_path
    )
    predictor = EyewearPredictor(model_path=model_path, model_type=args.model_type, confidence_threshold=0.5)

    df = pd.read_csv(args.val_csv)
    y_true: list[int] = []
    y_score: list[float] = []
    for row in df.itertuples(index=False):
        img = cv2.imread(str(row.image_path))
        if img is None:
            continue
        preds = predictor.predict_image(img)
        if not preds:
            continue
        top = max(preds, key=lambda x: x.confidence)
        y_true.append(int(row.label))
        y_score.append(float(top.score_positive))

    if not y_true:
        raise ValueError("No predictions generated for threshold calibration.")
    y_true_np = np.array(y_true)
    y_score_np = np.array(y_score)
    best_thr, best_f1 = best_threshold_by_f1(y_true_np, y_score_np)
    save_threshold(args.output, best_thr, metric_name="f1")
    print(f"Best threshold={best_thr:.3f} | F1={best_f1:.4f} | saved={args.output}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import ProjectConfig, ensure_dirs
from .inference import EyewearPredictor
from .metrics import classification_metrics, save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate eyewear model")
    parser.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--model-type", choices=["baseline", "torch"], default="baseline")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    ensure_dirs(cfg)
    model_path = args.model_path or (
        cfg.baseline_model_path if args.model_type == "baseline" else cfg.torch_model_path
    )
    predictor = EyewearPredictor(
        model_path=model_path,
        model_type=args.model_type,
        confidence_threshold=args.threshold,
    )

    df = pd.read_csv(args.test_csv)
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    for row in df.itertuples(index=False):
        image = cv2.imread(str(row.image_path))
        if image is None:
            continue
        preds = predictor.predict_image(image)
        if not preds:
            continue
        top = max(preds, key=lambda x: x.confidence)
        y_true.append(int(row.label))
        y_pred.append(top.label_id)
        y_score.append(top.score_positive)

    if not y_true:
        raise ValueError("Evaluation failed: no valid predictions were produced.")

    yt = np.array(y_true)
    yp = np.array(y_pred)
    ys = np.array(y_score)
    metrics = classification_metrics(yt, yp, ys)

    save_confusion_matrix(yt, yp, cfg.reports_dir / "confusion_matrix.png")
    report = {
        "model_type": args.model_type,
        "model_path": str(model_path),
        "threshold": args.threshold,
        "samples": len(y_true),
        **metrics,
    }
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

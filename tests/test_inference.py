from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from eyewear_pipeline.inference import EyewearPredictor


def test_predict_image_returns_predictions(trained_baseline_model: Path) -> None:
    predictor = EyewearPredictor(model_path=trained_baseline_model, model_type="baseline")
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    cv2.circle(image, (32, 36), 8, (0, 0, 0), 2)
    cv2.circle(image, (64, 36), 8, (0, 0, 0), 2)
    cv2.line(image, (40, 36), (56, 36), (0, 0, 0), 2)
    preds = predictor.predict_image(image)
    assert len(preds) >= 1
    assert preds[0].label_name in {"glasses", "no_glasses"}
    assert 0.0 <= preds[0].confidence <= 1.0


from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .calibration import best_threshold_by_f1
from .inference import EyewearPredictor
from .metrics import classification_metrics, save_confusion_matrix


@dataclass(slots=True)
class ModelMetadata:
    model_type: str
    model_path: str
    confidence_threshold: float
    threshold_version: str = "manual"
    model_version: str = "dev"
    registry_stage: str = "local"
    dataset_version: str = "unknown"
    run_id: str | None = None
    registered_model: str | None = None
    source_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_model_metadata(path: Path) -> ModelMetadata | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ModelMetadata(
        model_type=payload["model_type"],
        model_path=payload["model_path"],
        confidence_threshold=float(payload["confidence_threshold"]),
        threshold_version=payload.get("threshold_version", "manual"),
        model_version=payload.get("model_version", "dev"),
        registry_stage=payload.get("registry_stage", "local"),
        dataset_version=payload.get("dataset_version", "unknown"),
        run_id=payload.get("run_id"),
        registered_model=payload.get("registered_model"),
        source_uri=payload.get("source_uri"),
    )


def write_model_metadata(path: Path, metadata: ModelMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")


def evaluate_model(
    *,
    test_csv: Path,
    model_path: Path,
    model_type: str,
    threshold: float,
    confusion_matrix_path: Path | None = None,
) -> dict[str, Any]:
    predictor = EyewearPredictor(
        model_path=model_path,
        model_type=model_type,
        confidence_threshold=threshold,
    )

    df = pd.read_csv(test_csv)
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    for row in df.itertuples(index=False):
        image = cv2.imread(str(row.image_path))
        if image is None:
            continue
        predictions = predictor.predict_image(image)
        if not predictions:
            continue
        top = max(predictions, key=lambda item: item.confidence)
        y_true.append(int(row.label))
        y_pred.append(top.label_id)
        y_score.append(top.score_positive)

    if not y_true:
        raise ValueError("Evaluation failed: no valid predictions were produced.")

    true_values = np.array(y_true)
    predicted_values = np.array(y_pred)
    scores = np.array(y_score)
    metrics = classification_metrics(true_values, predicted_values, scores)

    if confusion_matrix_path is not None:
        save_confusion_matrix(true_values, predicted_values, confusion_matrix_path)

    return {
        "model_type": model_type,
        "model_path": str(model_path),
        "threshold": threshold,
        "samples": len(y_true),
        **metrics,
    }


def calibrate_threshold_from_csv(
    *,
    val_csv: Path,
    model_path: Path,
    model_type: str,
) -> tuple[float, float]:
    predictor = EyewearPredictor(model_path=model_path, model_type=model_type, confidence_threshold=0.5)
    df = pd.read_csv(val_csv)
    y_true: list[int] = []
    y_score: list[float] = []

    for row in df.itertuples(index=False):
        image = cv2.imread(str(row.image_path))
        if image is None:
            continue
        predictions = predictor.predict_image(image)
        if not predictions:
            continue
        top = max(predictions, key=lambda item: item.confidence)
        y_true.append(int(row.label))
        y_score.append(float(top.score_positive))

    if not y_true:
        raise ValueError("No predictions generated for threshold calibration.")

    return best_threshold_by_f1(np.array(y_true), np.array(y_score))

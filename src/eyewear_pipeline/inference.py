from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np

from .baseline import extract_features_bgr
from .data import ID_TO_LABEL
from .face import HaarFaceDetector
from .torch_model import predict_torch_positive_score


@dataclass(slots=True)
class Prediction:
    label_id: int
    label_name: str
    confidence: float
    score_positive: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(slots=True)
class _FallbackFaceBox:
    w: int
    h: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (0, 0, self.w, self.h)


class EyewearPredictor:
    def __init__(
        self,
        model_path: Path,
        model_type: str = "baseline",
        haar_cascade_path: str | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.face_detector = HaarFaceDetector(cascade_path=haar_cascade_path)
        self.model = None
        if self.model_type == "baseline":
            self.model = joblib.load(model_path)["pipeline"]

    def _predict_crop(self, face_crop_bgr: np.ndarray) -> tuple[int, float, float]:
        if self.model_type == "baseline":
            feats = extract_features_bgr(face_crop_bgr).reshape(1, -1)
            probs = self.model.predict_proba(feats)[0]
            positive_score = float(probs[1])
            label = int(positive_score >= self.confidence_threshold)
            conf = positive_score if label == 1 else 1.0 - positive_score
            return label, conf, positive_score
        if self.model_type == "torch":
            positive_score = predict_torch_positive_score(self.model_path, face_crop_bgr)
            label = int(positive_score >= self.confidence_threshold)
            conf = positive_score if label == 1 else 1.0 - positive_score
            return label, conf, positive_score
        raise ValueError(f"Unsupported model_type={self.model_type}")

    def predict_image(self, image_bgr: np.ndarray) -> list[Prediction]:
        h, w = image_bgr.shape[:2]
        faces = self.face_detector.detect(image_bgr)
        if not faces:
            faces = [_FallbackFaceBox(w=w, h=h)]

        results: list[Prediction] = []
        for box in faces:
            x1, y1, x2, y2 = box.as_xyxy()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            label_id, conf, score_positive = self._predict_crop(crop)
            results.append(
                Prediction(
                    label_id=label_id,
                    label_name=ID_TO_LABEL.get(label_id, "unknown"),
                    confidence=conf,
                    score_positive=score_positive,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return results


def draw_predictions(frame_bgr: np.ndarray, predictions: list[Prediction]) -> np.ndarray:
    vis = frame_bgr.copy()
    for pred in predictions:
        x1, y1, x2, y2 = pred.bbox_xyxy
        color = (0, 200, 0) if pred.label_id == 1 else (0, 120, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{pred.label_name}: {pred.confidence:.2f}"
        cv2.putText(vis, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis

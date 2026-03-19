from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_features_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    edges = cv2.Canny(gray, 80, 180)
    edge_density = edges.mean() / 255.0
    flat = resized.astype(np.float32).flatten() / 255.0
    return np.concatenate([flat, hist, np.array([edge_density], dtype=np.float32)])


@dataclass(slots=True)
class BaselineArtifact:
    pipeline: Pipeline
    model_type: str = "baseline"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_type": self.model_type, "pipeline": self.pipeline}, path)


def train_baseline(train_csv: Path) -> BaselineArtifact:
    df = pd.read_csv(train_csv)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for row in df.itertuples(index=False):
        img = cv2.imread(str(row.image_path))
        if img is None:
            continue
        features.append(extract_features_bgr(img))
        labels.append(int(row.label))
    if not features:
        raise ValueError("No valid images loaded for baseline training.")

    X = np.stack(features)
    y = np.array(labels)
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X, y)
    return BaselineArtifact(pipeline=pipeline)


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def best_threshold_by_f1(y_true: np.ndarray, y_score_positive: np.ndarray) -> tuple[float, float]:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (y_score_positive >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr, best_f1


def save_threshold(path: Path, threshold: float, metric_name: str = "f1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"threshold": float(threshold), "selection_metric": metric_name}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_threshold(path: Path, default: float = 0.5) -> float:
    if not path.exists():
        return default
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("threshold", default))


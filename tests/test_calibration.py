from __future__ import annotations

import numpy as np

from eyewear_pipeline.calibration import best_threshold_by_f1


def test_best_threshold_by_f1_returns_valid_values() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_score = np.array([0.1, 0.2, 0.7, 0.9, 0.8, 0.3, 0.6, 0.4])
    thr, f1 = best_threshold_by_f1(y_true, y_score)
    assert 0.05 <= thr <= 0.95
    assert 0.0 <= f1 <= 1.0


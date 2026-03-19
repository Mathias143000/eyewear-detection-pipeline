from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from eyewear_pipeline.baseline import train_baseline


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    (raw / "glasses").mkdir(parents=True)
    (raw / "no_glasses").mkdir(parents=True)

    for i in range(4):
        img_g = np.full((64, 64, 3), 150, dtype=np.uint8)
        cv2.circle(img_g, (24, 28), 7, (0, 0, 0), 2)
        cv2.circle(img_g, (40, 28), 7, (0, 0, 0), 2)
        cv2.line(img_g, (31, 28), (33, 28), (0, 0, 0), 2)
        cv2.imwrite(str(raw / "glasses" / f"g_{i}.png"), img_g)

        img_n = np.full((64, 64, 3), 150, dtype=np.uint8)
        cv2.circle(img_n, (24, 28), 2, (0, 0, 0), -1)
        cv2.circle(img_n, (40, 28), 2, (0, 0, 0), -1)
        cv2.imwrite(str(raw / "no_glasses" / f"n_{i}.png"), img_n)

    train_rows = []
    for p in (raw / "glasses").glob("*.png"):
        train_rows.append({"image_path": str(p), "label": 1})
    for p in (raw / "no_glasses").glob("*.png"):
        train_rows.append({"image_path": str(p), "label": 0})

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(train_rows).to_csv(csv_path, index=False)
    return raw, csv_path


@pytest.fixture()
def trained_baseline_model(tmp_path: Path, tiny_dataset: tuple[Path, Path]) -> Path:
    _, csv_path = tiny_dataset
    artifact = train_baseline(csv_path)
    model_path = tmp_path / "baseline.joblib"
    artifact.save(model_path)
    return model_path


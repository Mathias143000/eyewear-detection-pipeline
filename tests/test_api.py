from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")


def test_health() -> None:
    from eyewear_pipeline.api.main import app

    client = testclient.TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_reads_threshold_from_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    threshold_file = tmp_path / "threshold.json"
    threshold_file.write_text('{"threshold": 0.73, "selection_metric": "f1"}', encoding="utf-8")
    monkeypatch.setenv("EYEWEAR_THRESHOLD_FILE", str(threshold_file))
    monkeypatch.delenv("EYEWEAR_CONFIDENCE_THRESHOLD", raising=False)

    from importlib import reload

    import eyewear_pipeline.api.main as api_main

    reload(api_main)
    client = testclient.TestClient(api_main.app)
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["confidence_threshold"] == "0.73"


def test_predict_image_endpoint(monkeypatch: pytest.MonkeyPatch, trained_baseline_model: Path, tmp_path: Path) -> None:
    monkeypatch.setenv("EYEWEAR_MODEL_TYPE", "baseline")
    monkeypatch.setenv("EYEWEAR_MODEL_PATH", str(trained_baseline_model))

    from importlib import reload

    import eyewear_pipeline.api.main as api_main

    reload(api_main)
    client = testclient.TestClient(api_main.app)

    img = np.full((96, 96, 3), 150, dtype=np.uint8)
    cv2.circle(img, (32, 36), 8, (0, 0, 0), 2)
    cv2.circle(img, (64, 36), 8, (0, 0, 0), 2)
    cv2.line(img, (40, 36), (56, 36), (0, 0, 0), 2)
    image_path = tmp_path / "sample.png"
    cv2.imwrite(str(image_path), img)

    with image_path.open("rb") as f:
        response = client.post("/predict/image", files={"file": ("sample.png", f, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body

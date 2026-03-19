from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from eyewear_pipeline.calibration import load_threshold
from eyewear_pipeline.config import ProjectConfig
from eyewear_pipeline.inference import EyewearPredictor

app = FastAPI(title="Eyewear Detection API", version="0.1.0")

cfg = ProjectConfig()
model_type = os.getenv("EYEWEAR_MODEL_TYPE", "baseline")
default_model = cfg.baseline_model_path if model_type == "baseline" else cfg.torch_model_path
model_path = Path(os.getenv("EYEWEAR_MODEL_PATH", str(default_model)))


def _resolve_threshold() -> float:
    if "EYEWEAR_CONFIDENCE_THRESHOLD" in os.environ:
        return float(os.environ["EYEWEAR_CONFIDENCE_THRESHOLD"])

    threshold_file_env = os.getenv("EYEWEAR_THRESHOLD_FILE")
    if threshold_file_env:
        return load_threshold(Path(threshold_file_env), default=cfg.confidence_threshold)

    if model_type == "torch" and cfg.torch_hf_threshold_file_path.exists():
        return load_threshold(cfg.torch_hf_threshold_file_path, default=cfg.confidence_threshold)
    return load_threshold(cfg.threshold_file_path, default=cfg.confidence_threshold)


threshold = _resolve_threshold()


def _build_predictor() -> EyewearPredictor:
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    return EyewearPredictor(
        model_path=model_path,
        model_type=model_type,
        haar_cascade_path=cfg.haar_cascade_path,
        confidence_threshold=threshold,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {
        "model_type": model_type,
        "model_path": str(model_path),
        "confidence_threshold": f"{threshold:.2f}",
        "threshold_source": os.getenv("EYEWEAR_THRESHOLD_FILE", "artifacts/threshold*.json"),
    }


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)) -> dict:
    try:
        raw = await file.read()
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image payload.")
        predictor = _build_predictor()
        preds = predictor.predict_image(image)
        return {
            "predictions": [
                {
                    "label_id": p.label_id,
                    "label_name": p.label_name,
                    "confidence": p.confidence,
                    "score_positive": p.score_positive,
                    "bbox_xyxy": list(p.bbox_xyxy),
                }
                for p in preds
            ]
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

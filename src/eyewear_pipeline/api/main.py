from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile

from eyewear_pipeline.config import ProjectConfig, ensure_dirs

from .observability import (
    observe_failure,
    observe_predictions,
    observe_request,
    render_metrics,
)
from .runtime import (
    build_model_info,
    configure_logging,
    create_runtime,
    emit_log,
    ensure_predictor_loaded,
    request_client_ip,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_state()
    logger = get_logger()
    emit_log(logger, event="app_started", service="eyewear-api")
    yield
    emit_log(logger, event="app_stopped", service="eyewear-api")


app = FastAPI(title="Eyewear Detection API", version="0.2.0", lifespan=lifespan)


def initialize_app_state() -> None:
    config = ProjectConfig()
    ensure_dirs(config)
    logger = configure_logging(config.log_path)
    runtime = create_runtime(config)
    ensure_predictor_loaded(runtime, logger)
    app.state.config = config
    app.state.logger = logger
    app.state.runtime = runtime


def get_runtime():
    if not hasattr(app.state, "runtime"):
        initialize_app_state()
    return app.state.runtime


def get_logger():
    if not hasattr(app.state, "logger"):
        initialize_app_state()
    return app.state.logger


initialize_app_state()


@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        observe_failure("request")
        raise
    finally:
        duration = perf_counter() - started
        observe_request(request.method, request.url.path, str(status_code), duration)
        emit_log(
            get_logger(),
            event="http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            client_ip=request_client_ip(request),
        )


@app.get("/health")
def health() -> dict[str, object]:
    runtime = get_runtime()
    ensure_predictor_loaded(runtime, get_logger())
    return {"status": "ok", "ready": runtime.predictor is not None}


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready(response: Response) -> dict[str, object]:
    runtime = get_runtime()
    ensure_predictor_loaded(runtime, get_logger())
    if runtime.predictor is None:
        response.status_code = 503
        return {"status": "not_ready", "detail": runtime.load_error}
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.get("/model-info")
def model_info() -> dict[str, object]:
    runtime = get_runtime()
    ensure_predictor_loaded(runtime, get_logger())
    return build_model_info(runtime)


@app.post("/predict/image")
async def predict_image(request: Request, file: UploadFile = File(...)) -> dict:
    runtime = get_runtime()
    ensure_predictor_loaded(runtime, get_logger())
    if runtime.predictor is None:
        raise HTTPException(status_code=503, detail=runtime.load_error or "Model is not ready.")

    try:
        raw = await file.read()
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image payload.")
        predictions = runtime.predictor.predict_image(image)
        observe_predictions(pred.label_name for pred in predictions)
        metadata = build_model_info(runtime)
        return {
            "request_id": request.state.request_id,
            "model": {
                "model_type": metadata["model_type"],
                "model_version": metadata.get("model_version", "dev"),
                "registry_stage": metadata.get("registry_stage", "local"),
                "confidence_threshold": metadata["confidence_threshold"],
            },
            "predictions": [
                {
                    "label_id": p.label_id,
                    "label_name": p.label_name,
                    "confidence": p.confidence,
                    "score_positive": p.score_positive,
                    "bbox_xyxy": list(p.bbox_xyxy),
                }
                for p in predictions
            ]
        }
    except HTTPException:
        raise
    except Exception as exc:
        observe_failure("predict")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

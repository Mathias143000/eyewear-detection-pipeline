from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from fastapi import Request

from eyewear_pipeline.calibration import load_threshold
from eyewear_pipeline.config import ProjectConfig
from eyewear_pipeline.inference import EyewearPredictor
from eyewear_pipeline.mlops import ModelMetadata, read_model_metadata

from .observability import set_model_info


LOGGER_NAME = "eyewear.api"


@dataclass(slots=True)
class RuntimeState:
    config: ProjectConfig
    model_type: str
    model_path: Path
    predictor: EyewearPredictor | None = None
    metadata: ModelMetadata | None = None
    signature: tuple[str, float, str, float, str] | None = None
    load_error: str | None = None


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def emit_log(logger: logging.Logger, **payload: Any) -> None:
    payload.setdefault("ts", round(time(), 3))
    logger.info(json.dumps(payload, ensure_ascii=False))


def create_runtime(config: ProjectConfig) -> RuntimeState:
    model_type = os.getenv("EYEWEAR_MODEL_TYPE", "baseline")
    default_model = config.baseline_model_path if model_type == "baseline" else config.torch_model_path
    model_path = Path(os.getenv("EYEWEAR_MODEL_PATH", str(default_model)))
    return RuntimeState(config=config, model_type=model_type, model_path=model_path)


def resolve_threshold(config: ProjectConfig, model_type: str, metadata: ModelMetadata | None) -> float:
    if "EYEWEAR_CONFIDENCE_THRESHOLD" in os.environ:
        return float(os.environ["EYEWEAR_CONFIDENCE_THRESHOLD"])

    threshold_file_env = os.getenv("EYEWEAR_THRESHOLD_FILE")
    if threshold_file_env:
        return load_threshold(Path(threshold_file_env), default=config.confidence_threshold)
    if metadata is not None:
        return float(metadata.confidence_threshold)
    if model_type == "torch" and config.torch_hf_threshold_file_path.exists():
        return load_threshold(config.torch_hf_threshold_file_path, default=config.confidence_threshold)
    return load_threshold(config.threshold_file_path, default=config.confidence_threshold)


def ensure_predictor_loaded(state: RuntimeState, logger: logging.Logger) -> None:
    metadata = read_model_metadata(state.config.model_metadata_path)
    threshold = resolve_threshold(state.config, state.model_type, metadata)
    metadata_path = state.config.model_metadata_path
    metadata_marker = str(metadata_path)
    metadata_mtime = metadata_path.stat().st_mtime if metadata_path.exists() else -1.0
    model_mtime = state.model_path.stat().st_mtime if state.model_path.exists() else -1.0
    signature = (str(state.model_path), model_mtime, metadata_marker, metadata_mtime, f"{threshold:.5f}")

    if state.predictor is not None and state.signature == signature:
        return

    if not state.model_path.exists():
        state.predictor = None
        state.metadata = metadata
        state.signature = signature
        state.load_error = f"Model file not found: {state.model_path}"
        set_model_info(
            model_type=state.model_type,
            model_version="missing",
            registry_stage=metadata.registry_stage if metadata else "local",
            threshold_version="missing",
        )
        emit_log(
            logger,
            event="predictor_not_ready",
            model_type=state.model_type,
            model_path=str(state.model_path),
            error=state.load_error,
        )
        return

    try:
        state.predictor = EyewearPredictor(
            model_path=state.model_path,
            model_type=state.model_type,
            haar_cascade_path=state.config.haar_cascade_path,
            confidence_threshold=threshold,
        )
        state.metadata = metadata
        state.signature = signature
        state.load_error = None
        set_model_info(
            model_type=state.model_type,
            model_version=metadata.model_version if metadata else "local",
            registry_stage=metadata.registry_stage if metadata else "local",
            threshold_version=metadata.threshold_version if metadata else "manual",
        )
        emit_log(
            logger,
            event="predictor_loaded",
            model_type=state.model_type,
            model_path=str(state.model_path),
            model_version=metadata.model_version if metadata else "local",
            threshold=threshold,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime branch
        state.predictor = None
        state.metadata = metadata
        state.signature = signature
        state.load_error = str(exc)
        set_model_info(
            model_type=state.model_type,
            model_version="error",
            registry_stage=metadata.registry_stage if metadata else "local",
            threshold_version="error",
        )
        emit_log(
            logger,
            event="predictor_load_failed",
            model_type=state.model_type,
            model_path=str(state.model_path),
            error=str(exc),
        )


def build_model_info(state: RuntimeState) -> dict[str, Any]:
    threshold = resolve_threshold(state.config, state.model_type, state.metadata)
    info: dict[str, Any] = {
        "model_type": state.model_type,
        "model_path": str(state.model_path),
        "confidence_threshold": f"{threshold:.2f}",
        "ready": state.predictor is not None,
        "threshold_source": os.getenv("EYEWEAR_THRESHOLD_FILE", str(state.config.model_metadata_path)),
    }
    if state.metadata is not None:
        info.update(
            {
                "model_version": state.metadata.model_version,
                "registry_stage": state.metadata.registry_stage,
                "dataset_version": state.metadata.dataset_version,
                "threshold_version": state.metadata.threshold_version,
                "run_id": state.metadata.run_id,
                "registered_model": state.metadata.registered_model,
            }
        )
    if state.load_error:
        info["load_error"] = state.load_error
    return info


def request_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

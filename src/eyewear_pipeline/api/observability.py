from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUESTS = Counter(
    "eyewear_inference_requests_total",
    "HTTP requests processed by the inference API.",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "eyewear_inference_request_duration_seconds",
    "Request latency for the inference API.",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "eyewear_predictions_total",
    "Predictions emitted by the inference API.",
    ["label_name"],
)
FAILURES = Counter(
    "eyewear_failures_total",
    "Failures observed by the inference API.",
    ["scope"],
)
MODEL_INFO = Gauge(
    "eyewear_model_info",
    "Static information about the currently served model.",
    ["model_type", "model_version", "registry_stage", "threshold_version"],
)


def observe_request(method: str, path: str, status: int, duration_seconds: float) -> None:
    REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration_seconds)


def observe_predictions(labels: list[str]) -> None:
    if not labels:
        return
    for label in labels:
        PREDICTIONS.labels(label_name=label).inc()


def observe_failure(scope: str) -> None:
    FAILURES.labels(scope=scope).inc()


def set_model_info(
    *,
    model_type: str,
    model_version: str,
    registry_stage: str,
    threshold_version: str,
) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(
        model_type=model_type or "unknown",
        model_version=model_version or "unknown",
        registry_stage=registry_stage or "unknown",
        threshold_version=threshold_version or "unknown",
    ).set(1)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

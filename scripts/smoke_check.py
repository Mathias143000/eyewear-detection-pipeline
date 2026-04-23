from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from mlflow import MlflowClient


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke checks for the local MLOps lab.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18040")
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:15040")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:19090")
    parser.add_argument("--grafana-url", default="http://127.0.0.1:13040")
    parser.add_argument("--loki-url", default="http://127.0.0.1:13100")
    parser.add_argument("--registered-model", default="eyewear-classifier")
    parser.add_argument("--sample-image", type=Path, default=None)
    return parser.parse_args()


def http_json(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def http_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def wait_until(check, timeout: int = 60, interval: float = 2.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            check()
            return
        except Exception as exc:  # pragma: no cover - runtime polling helper
            last_error = exc
            time.sleep(interval)
    if last_error is not None:
        raise last_error
    raise TimeoutError("Timed out waiting for check.")


def resolve_sample_image(sample_image: Path | None) -> Path:
    if sample_image is not None:
        return sample_image
    candidates = sorted((ROOT / "data" / "raw").glob("*/*.png"))
    if not candidates:
        raise FileNotFoundError("No sample image found under data/raw. Run prepare_demo_data first.")
    return candidates[0]


def predict(base_url: str, image_path: Path) -> dict:
    boundary = "----eyewearboundary"
    file_bytes = image_path.read_bytes()
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    return http_json(f"{base_url}/predict/image", method="POST", body=payload, headers=headers)


def main() -> None:
    args = parse_args()
    sample_image = resolve_sample_image(args.sample_image)

    wait_until(lambda: http_json(f"{args.base_url}/ready"))
    health = http_json(f"{args.base_url}/health")
    ready = http_json(f"{args.base_url}/ready")
    model_info = http_json(f"{args.base_url}/model-info")
    prediction = predict(args.base_url, sample_image)
    metrics_text = http_text(f"{args.base_url}/metrics")

    client = MlflowClient(tracking_uri=args.mlflow_uri)
    registered_model = client.get_registered_model(args.registered_model)
    targets = http_json(f"{args.prometheus_url}/api/v1/targets")
    grafana_health = http_json(f"{args.grafana_url}/api/health")

    def query_loki() -> dict:
        encoded = urllib.parse.quote('{job="eyewear-api"}')
        return http_json(f"{args.loki_url}/loki/api/v1/query?query={encoded}")

    wait_until(lambda: query_loki()["data"]["result"])
    loki_result = query_loki()

    summary = {
        "health": health,
        "ready": ready,
        "model_info": model_info,
        "prediction_count": len(prediction.get("predictions", [])),
        "request_id": prediction.get("request_id"),
        "metrics_ok": "eyewear_inference_requests_total" in metrics_text,
        "registered_model": args.registered_model,
        "latest_versions": [version.version for version in registered_model.latest_versions],
        "prometheus_targets": len(targets["data"]["activeTargets"]),
        "grafana_status": grafana_health["database"],
        "loki_streams": len(loki_result["data"]["result"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

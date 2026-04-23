# Architecture

## Portfolio Role

`eyewear-detection-pipeline` is the MLOps workload in the portfolio.
It demonstrates how a small CV model can be wrapped in production-style delivery,
tracking, promotion, and observability flows without pretending to be a full research platform.

## Runtime Topology

```text
client
  -> nginx edge
  -> FastAPI inference API
     -> baseline model artifact + threshold metadata

training / promotion scripts
  -> MLflow tracking server
     -> PostgreSQL backend store
     -> MinIO artifact store

observability
  -> Prometheus scrapes /metrics
  -> Loki receives API logs via Promtail
  -> Grafana visualizes metrics + logs
```

## Workload Flow

1. `prepare_demo_data.py` creates a deterministic synthetic dataset and split manifests.
2. `train_smoke.py` trains the baseline model, calibrates threshold, evaluates quality, and logs the run to MLflow.
3. `register_model.py` creates a model version in the MLflow registry.
4. `promote_model.py` marks the model as production, downloads the serving artifact, and writes `serving_model.json`.
5. The running API lazily reloads the promoted model and threshold on the next request.
6. `smoke_check.py` validates the full path through `nginx`, MLflow, Prometheus, Grafana, and Loki.

## Deliberate Scope Choices

- The promoted demo path is `baseline` and CPU-first, so the lab remains reproducible on a regular laptop.
- `torch` training and heavier experiment paths still exist in the repo, but they are not the required smoke path.
- The API does not depend on MLflow to answer inference requests after promotion; it serves a local promoted artifact.
- Logs are scraped from the shared `logs/` mount instead of the Docker daemon, which keeps the local lab simpler on Windows.

## Known Constraints

- The default smoke dataset is synthetic and intended for reproducibility, not for real-world quality claims.
- Model registry stages in MLflow are used because they are easy to demonstrate, even though the MLflow project has started deprecating stage-centric workflows.
- The local compose stack is intentionally single-node and not meant to simulate GPU scheduling or distributed training.

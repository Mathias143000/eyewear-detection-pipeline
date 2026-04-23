# Eyewear Detection Pipeline

`eyewear-detection-pipeline` is the MLOps showcase in the portfolio.
It turns a small CV classifier into a production-style demo stand with:

- reproducible demo data preparation
- baseline training, evaluation, and threshold calibration
- MLflow tracking + model registry
- MinIO artifact storage
- promoted local serving artifact for the inference API
- `nginx` edge in front of FastAPI
- Prometheus, Grafana, Loki, and Promtail for runtime visibility
- repeatable smoke and evidence collection scripts

## What This Repo Demonstrates

- how a model is trained, tracked, registered, promoted, and served
- how ML artifacts move from experiment tracking to a promoted serving payload
- how to expose service health, readiness, liveness, metrics, and structured logs
- how to wrap an ML workload in a portfolio-ready local lab instead of a notebook-only project

## Stack

- `FastAPI` inference API
- `nginx` edge proxy
- `MLflow` tracking + registry
- `MinIO` artifact store
- `PostgreSQL` backend for MLflow metadata
- `Prometheus` metrics
- `Grafana` dashboards
- `Loki + Promtail` log pipeline
- Python operator scripts for demo data, training, promotion, smoke, and evidence

## Quick Demo

The fastest repeatable path is:

```powershell
python scripts/bootstrap_env.py demo
```

That single command:

1. starts the compose lab
2. prepares deterministic synthetic data
3. trains and tracks the baseline model
4. registers and promotes the model
5. validates inference through the edge layer
6. collects evidence under `artifacts/evidence`

## Manual Demo Flow

```powershell
docker compose up -d --build

$env:MLFLOW_TRACKING_URI='http://127.0.0.1:15040'
$env:MLFLOW_S3_ENDPOINT_URL='http://127.0.0.1:19140'
$env:AWS_ACCESS_KEY_ID='minioadmin'
$env:AWS_SECRET_ACCESS_KEY='minioadmin123'
$env:AWS_DEFAULT_REGION='us-east-1'

python scripts/prepare_demo_data.py --force
python scripts/train_smoke.py --tracking-uri http://127.0.0.1:15040
python scripts/register_model.py --tracking-uri http://127.0.0.1:15040
python scripts/promote_model.py --tracking-uri http://127.0.0.1:15040
python scripts/smoke_check.py
python scripts/collect_evidence.py
```

## Service Endpoints

- edge: `http://127.0.0.1:18040`
- MLflow: `http://127.0.0.1:15040`
- MinIO API: `http://127.0.0.1:19140`
- MinIO console: `http://127.0.0.1:19141`
- Prometheus: `http://127.0.0.1:19090`
- Grafana: `http://127.0.0.1:13040`
- Loki: `http://127.0.0.1:13100`

Grafana default credentials:

- user: `admin`
- password: `admin`

## Important API Routes

- `GET /health`
- `GET /live`
- `GET /ready`
- `GET /metrics`
- `GET /model-info`
- `POST /predict/image`

## Portfolio-Ready Evidence

This repo now includes the pieces that make it useful as a portfolio artifact, not just as code:

- clean quick-start in this README
- architecture note in `docs/architecture.md`
- repeatable demo runbook in `runbooks/demo.md`
- CI flow that validates lint, tests, and the minimal ML promotion loop
- evidence bundle under `artifacts/evidence`

## Validation

The current `DoD` was validated with:

- `python -m ruff check .`
- `python -m pytest -q`
- `python scripts/train_smoke.py --prepare-data --tracking-uri file:./mlruns`
- `python scripts/register_model.py --tracking-uri file:./mlruns`
- `python scripts/promote_model.py --tracking-uri file:./mlruns`
- `docker compose up -d --build`
- `python scripts/train_smoke.py --tracking-uri http://127.0.0.1:15040`
- `python scripts/register_model.py --tracking-uri http://127.0.0.1:15040`
- `python scripts/promote_model.py --tracking-uri http://127.0.0.1:15040`
- `python scripts/smoke_check.py`
- `python scripts/collect_evidence.py`

## Known Limitations

- The guaranteed smoke path is `baseline` and CPU-first. The heavier `torch` path still exists, but it is not the default promoted flow.
- The synthetic dataset is for reproducibility and demo stability, not for real-world quality claims.
- The current lab focuses on metrics and logs. Alerting, tracing, drift analysis, ONNX export, and GPU profiles are outside this portfolio slice.
- If `edge` returns `502` after recreating the API container, restarting `edge` refreshes the upstream IP. This is documented in the runbook.

## Explicit Non-Goals

- add Alertmanager and alert rules
- add drift and quality regression reports
- add ONNX export and CPU benchmark comparison
- add signed images / SBOM / dependency scan in CI
- add optional GPU profile for the torch path

These are deliberate MLOps extension points, not unfinished DoD work. The current repo is complete as a reproducible MLflow + MinIO + FastAPI serving showcase.

# Demo Runbook

## Fastest Demo

```powershell
python scripts/bootstrap_env.py demo
```

This command:

1. builds and starts the compose lab
2. prepares deterministic demo data
3. trains and tracks a baseline model in MLflow
4. registers and promotes the model
5. validates the full inference path
6. collects evidence under `artifacts/evidence`

## Manual Flow

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

## What To Show In An Interview

- MLflow experiment and registered model version
- `serving_model.json` proving which run is currently served
- Grafana dashboard with latency, throughput, prediction labels, and logs
- `/model-info` showing promoted model metadata through the edge layer
- `artifacts/evidence` as portable proof of the run

## Troubleshooting

- If `edge` returns `502` after recreating the API container, restart `edge` so `nginx` refreshes the upstream IP.
- If training cannot upload artifacts, verify `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`.
- If `/ready` returns `503`, confirm that `models/baseline_glasses.joblib`, `artifacts/threshold.json`, and `artifacts/serving_model.json` exist.
- If Grafana has no logs, generate one prediction request and then rerun `smoke_check.py` after a few seconds.

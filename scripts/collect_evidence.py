from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect lightweight evidence from the local MLOps lab.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18040")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:19090")
    parser.add_argument("--grafana-url", default="http://127.0.0.1:13040")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "evidence")
    return parser.parse_args()


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=20) as response:
        destination.write_bytes(response.read())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fetch(f"{args.base_url}/model-info", args.output_dir / "model-info.json")
    fetch(f"{args.base_url}/metrics", args.output_dir / "metrics.txt")
    fetch(f"{args.prometheus_url}/api/v1/targets", args.output_dir / "prometheus-targets.json")
    fetch(f"{args.grafana_url}/api/health", args.output_dir / "grafana-health.json")

    for artifact_name in (
        "dataset_manifest.json",
        "train_smoke_run.json",
        "registered_model.json",
        "promotion.json",
        "serving_model.json",
    ):
        source = ROOT / "artifacts" / artifact_name
        if source.exists():
            shutil.copy2(source, args.output_dir / artifact_name)

    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    for report_name in ("metrics.json", "confusion_matrix.png"):
        source = ROOT / "reports" / report_name
        if source.exists():
            shutil.copy2(source, reports_dir / report_name)

    summary = {"evidence_dir": str(args.output_dir)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

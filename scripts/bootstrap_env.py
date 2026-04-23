"""Bootstrap and validate the local MLOps lab."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:15040")
    env.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:19140")
    env.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    env.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin123")
    env.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    return env


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=ROOT, env=env or base_env())


def cmd_demo() -> None:
    env = base_env()
    run(["docker", "compose", "up", "-d", "--build"], env=env)
    run([sys.executable, "scripts/prepare_demo_data.py", "--force"], env=env)
    run([sys.executable, "scripts/train_smoke.py", "--tracking-uri", env["MLFLOW_TRACKING_URI"]], env=env)
    run([sys.executable, "scripts/register_model.py", "--tracking-uri", env["MLFLOW_TRACKING_URI"]], env=env)
    run([sys.executable, "scripts/promote_model.py", "--tracking-uri", env["MLFLOW_TRACKING_URI"]], env=env)
    run([sys.executable, "scripts/smoke_check.py"], env=env)
    run([sys.executable, "scripts/collect_evidence.py"], env=env)


def cmd_up() -> None:
    run(["docker", "compose", "up", "-d", "--build"])


def cmd_down() -> None:
    run(["docker", "compose", "down", "-v", "--remove-orphans"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap helper for the local eyewear MLOps lab.")
    parser.add_argument("command", choices=["up", "down", "demo"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "up":
        cmd_up()
    elif args.command == "down":
        cmd_down()
    else:
        cmd_demo()


if __name__ == "__main__":
    main()

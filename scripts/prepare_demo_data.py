"""Prepare deterministic demo data for the local MLOps lab."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic synthetic demo data.")
    parser.add_argument("--samples-per-class", type=int, default=48)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--metadata-output", type=Path, default=ROOT / "artifacts" / "dataset_manifest.json")
    parser.add_argument("--dataset-version", default="synthetic-smoke-v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_script(script_name: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / "scripts" / script_name), *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def main() -> None:
    args = parse_args()

    if args.force:
        for path in (args.dataset_root, args.splits_dir):
            if path.exists():
                shutil.rmtree(path)
        if args.manifest.exists():
            args.manifest.unlink()

    run_script(
        "create_synthetic_dataset.py",
        "--output-dir",
        str(args.dataset_root),
        "--samples-per-class",
        str(args.samples_per_class),
    )
    run_script(
        "prepare_data.py",
        "--dataset-root",
        str(args.dataset_root),
        "--manifest",
        str(args.manifest),
        "--splits-dir",
        str(args.splits_dir),
    )

    manifest_df = pd.read_csv(args.manifest)
    metadata = {
        "dataset_version": args.dataset_version,
        "manifest_path": str(args.manifest),
        "splits_dir": str(args.splits_dir),
        "samples_total": int(len(manifest_df)),
        "class_distribution": {
            str(int(label)): int(count)
            for label, count in manifest_df["label"].value_counts().sort_index().items()
        },
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

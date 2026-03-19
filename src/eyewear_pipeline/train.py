from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import train_baseline
from .config import ProjectConfig, ensure_dirs
from .torch_model import train_torch_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train eyewear model")
    parser.add_argument("--train-csv", type=Path, default=Path("data/splits/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/splits/val.csv"))
    parser.add_argument("--model-type", choices=["baseline", "torch"], default="baseline")
    parser.add_argument("--model-name", default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    ensure_dirs(cfg)

    if args.model_type == "baseline":
        artifact = train_baseline(args.train_csv)
        artifact.save(cfg.baseline_model_path)
        report = {"model_type": "baseline", "model_path": str(cfg.baseline_model_path)}
    else:
        result = train_torch_model(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            model_path=cfg.torch_model_path,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        report = {"model_type": "torch", "model_path": str(cfg.torch_model_path), **result}

    report_path = cfg.artifacts_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


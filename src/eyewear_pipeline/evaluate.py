from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProjectConfig, ensure_dirs
from .mlops import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate eyewear model")
    parser.add_argument("--test-csv", type=Path, default=Path("data/splits/test.csv"))
    parser.add_argument("--model-type", choices=["baseline", "torch"], default="baseline")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    ensure_dirs(cfg)
    model_path = args.model_path or (
        cfg.baseline_model_path if args.model_type == "baseline" else cfg.torch_model_path
    )
    report = evaluate_model(
        test_csv=args.test_csv,
        model_path=model_path,
        model_type=args.model_type,
        threshold=args.threshold,
        confusion_matrix_path=cfg.reports_dir / "confusion_matrix.png",
    )
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

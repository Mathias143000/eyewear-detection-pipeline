from __future__ import annotations

import argparse
from pathlib import Path

from eyewear_pipeline.data import build_manifest, split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manifest and splits")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_manifest(args.dataset_root, args.manifest)
    train_df, val_df, test_df = split_manifest(args.manifest, args.splits_dir)
    print(
        f"Manifest: {len(df)} rows | train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
    )


if __name__ == "__main__":
    main()


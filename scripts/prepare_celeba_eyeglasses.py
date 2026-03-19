from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eyewear_pipeline.data import split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CelebA manifest for eyeglasses classification")
    parser.add_argument("--images-dir", type=Path, required=True, help="Path to img_align_celeba directory")
    parser.add_argument("--attr-file", type=Path, required=True, help="Path to list_attr_celeba.txt")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_celeba.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=0,
        help="Optional cap per class (0 = no cap).",
    )
    return parser.parse_args()


def load_celeba_attributes(attr_file: Path) -> pd.DataFrame:
    raw = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)
    if "Eyeglasses" not in raw.columns:
        raise ValueError("Column 'Eyeglasses' not found in attribute file.")
    if "image_id" not in raw.columns:
        raw = raw.rename(columns={raw.columns[0]: "image_id"})
    return raw[["image_id", "Eyeglasses"]]


def main() -> None:
    args = parse_args()
    attrs = load_celeba_attributes(args.attr_file)
    attrs["label"] = (attrs["Eyeglasses"] == 1).astype(int)
    attrs["image_path"] = attrs["image_id"].map(lambda x: str(args.images_dir / x))
    attrs = attrs[attrs["image_path"].map(lambda p: Path(p).exists())].copy()

    if args.max_samples_per_class > 0:
        attrs = (
            attrs.groupby("label", group_keys=False)
            .head(args.max_samples_per_class)
            .reset_index(drop=True)
        )

    manifest = attrs[["image_path", "label"]]
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    train_df, val_df, test_df = split_manifest(args.manifest, args.splits_dir)
    print(
        f"CelebA manifest ready: total={len(manifest)} train={len(train_df)} "
        f"val={len(val_df)} test={len(test_df)}"
    )


if __name__ == "__main__":
    main()


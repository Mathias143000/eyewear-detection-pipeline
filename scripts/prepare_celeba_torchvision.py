from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torchvision.datasets import CelebA

from eyewear_pipeline.data import split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CelebA via torchvision and prepare eyeglasses manifest/splits"
    )
    parser.add_argument("--root", type=Path, default=Path("data/external"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_celeba.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--max-samples-per-class", type=int, default=2000)
    return parser.parse_args()


def _to_dataframe(ds: CelebA) -> pd.DataFrame:
    attr_names = list(ds.attr_names)
    eyeglasses_idx = attr_names.index("Eyeglasses")
    rows: list[dict[str, str | int]] = []
    for idx, filename in enumerate(ds.filename):
        label = int(ds.attr[idx][eyeglasses_idx].item() == 1)
        rows.append({"image_path": str(ds.root / ds.base_folder / "img_align_celeba" / filename), "label": label})
    return pd.DataFrame(rows)


def _cap_per_class(df: pd.DataFrame, max_per_class: int) -> pd.DataFrame:
    if max_per_class <= 0:
        return df
    return df.groupby("label", group_keys=False).head(max_per_class).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    train_ds = CelebA(root=str(args.root), split="train", target_type="attr", download=True)
    valid_ds = CelebA(root=str(args.root), split="valid", target_type="attr", download=False)
    test_ds = CelebA(root=str(args.root), split="test", target_type="attr", download=False)

    train_df = _cap_per_class(_to_dataframe(train_ds), args.max_samples_per_class)
    valid_df = _cap_per_class(_to_dataframe(valid_ds), max(300, args.max_samples_per_class // 4))
    test_df = _cap_per_class(_to_dataframe(test_ds), max(300, args.max_samples_per_class // 4))

    manifest = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)

    split_manifest(args.manifest, args.splits_dir, val_size=0.15, test_size=0.15, random_state=42)
    print(
        "CelebA prepared via torchvision: "
        f"manifest={len(manifest)} train={len(train_df)} valid={len(valid_df)} test={len(test_df)}"
    )


if __name__ == "__main__":
    main()


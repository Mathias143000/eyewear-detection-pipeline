from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from eyewear_pipeline.data import split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare eyeglasses dataset from Hugging Face CelebA")
    parser.add_argument("--repo-id", default="flwrlabs/celeba")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("data/hf_celeba"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest_hf_celeba.csv"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits_hf"))
    parser.add_argument("--max-samples-per-class", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    glasses_dir = args.output_dir / "glasses"
    no_glasses_dir = args.output_dir / "no_glasses"
    glasses_dir.mkdir(parents=True, exist_ok=True)
    no_glasses_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.repo_id, split=args.split)
    ds = ds.shuffle(seed=args.seed)

    # Column names in flwrlabs/celeba contain Eyeglasses directly.
    rows: list[dict[str, str | int]] = []
    class_counts = {0: 0, 1: 0}
    max_per_class = max(args.max_samples_per_class, 1)

    for idx, item in enumerate(ds):
        if "Eyeglasses" not in item:
            continue
        label = 1 if int(item["Eyeglasses"]) == 1 else 0
        if class_counts[label] >= max_per_class:
            if class_counts[0] >= max_per_class and class_counts[1] >= max_per_class:
                break
            continue

        image = item["image"]
        out_name = f"{'g' if label == 1 else 'n'}_{idx:06d}.jpg"
        out_path = (glasses_dir if label == 1 else no_glasses_dir) / out_name
        image.save(out_path)

        rows.append({"image_path": str(out_path), "label": label})
        class_counts[label] += 1

    if not rows:
        raise ValueError("No samples extracted from HF CelebA dataset.")

    manifest_df = pd.DataFrame(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.manifest, index=False)
    train_df, val_df, test_df = split_manifest(args.manifest, args.splits_dir)
    print(
        f"HF CelebA manifest ready: total={len(manifest_df)} "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"class0={class_counts[0]} class1={class_counts[1]}"
    )


if __name__ == "__main__":
    main()

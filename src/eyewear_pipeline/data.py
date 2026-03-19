from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_TO_ID = {"no_glasses": 0, "glasses": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


@dataclass(slots=True)
class Sample:
    image_path: Path
    label: int


def _iter_images(root: Path) -> Iterable[Path]:
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        yield from root.rglob(ext)


def collect_samples(dataset_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for class_name, class_id in LABEL_TO_ID.items():
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for image_path in _iter_images(class_dir):
            samples.append(Sample(image_path=image_path, label=class_id))
    return samples


def build_manifest(dataset_root: Path, output_path: Path) -> pd.DataFrame:
    samples = collect_samples(dataset_root)
    rows = [{"image_path": str(sample.image_path), "label": sample.label} for sample in samples]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found at {dataset_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def split_manifest(
    manifest_path: Path,
    output_dir: Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(manifest_path)
    if "label" not in df.columns:
        raise ValueError("Manifest must contain 'label' column.")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    return train_df, val_df, test_df


from __future__ import annotations

from pathlib import Path

from eyewear_pipeline.data import build_manifest, split_manifest


def test_build_manifest_and_split(tiny_dataset: tuple[Path, Path], tmp_path: Path) -> None:
    raw, _ = tiny_dataset
    manifest = tmp_path / "manifest.csv"
    df = build_manifest(raw, manifest)
    assert len(df) == 8
    assert manifest.exists()

    split_dir = tmp_path / "splits"
    train_df, val_df, test_df = split_manifest(manifest, split_dir, val_size=0.25, test_size=0.25)
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert (split_dir / "train.csv").exists()


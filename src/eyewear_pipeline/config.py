from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectConfig:
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    confidence_threshold: float = float(os.getenv("EYEWEAR_CONFIDENCE_THRESHOLD", "0.50"))
    threshold_file_path: Path = Path(os.getenv("EYEWEAR_THRESHOLD_FILE", "artifacts/threshold.json"))
    haar_cascade_path: str | None = os.getenv("EYEWEAR_HAAR_CASCADE")

    @property
    def baseline_model_path(self) -> Path:
        return self.models_dir / "baseline_glasses.joblib"

    @property
    def torch_model_path(self) -> Path:
        return self.models_dir / "mobilenet_glasses.pt"

    @property
    def torch_hf_threshold_file_path(self) -> Path:
        return self.artifacts_dir / "threshold_torch_hf.json"


def ensure_dirs(cfg: ProjectConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

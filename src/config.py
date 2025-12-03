"""Global configuration for the Automated Insight Engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used across the project."""

    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = root / "data"
    reports_dir: Path = root / "reports"
    models_dir: Path = root / "models"
    artifacts_dir: Path = root / "artifacts"
    uploads_dir: Path = root / "uploads"


@dataclass(frozen=True)
class ModelConfig:
    """Model training hyper-parameters."""

    target_column: str = "approved_conversion"
    numeric_features: List[str] = field(
        default_factory=lambda: ["impressions", "clicks", "spent", "total_conversion", "interest"]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: ["age", "gender", "xyz_campaign_id", "fb_campaign_id"]
    )
    test_size: float = 0.25
    random_state: int = 42


PATHS = Paths()
MODEL_CONFIG = ModelConfig()


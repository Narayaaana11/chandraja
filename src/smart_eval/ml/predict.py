from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


@dataclass
class TrainedScorePredictor:
    """Wrapper for inference with optional preprocessing from saved artifacts."""

    model: object
    preprocessor: object | None = None

    @classmethod
    def from_file(cls, model_path: str | Path) -> Optional["TrainedScorePredictor"]:
        """Load a predictor from a joblib artifact path."""
        path = Path(model_path)
        if not path.exists():
            return None

        artifact = joblib.load(path)
        model = artifact.get("model") if isinstance(artifact, dict) else artifact
        preprocessor = artifact.get("preprocessor") if isinstance(artifact, dict) else None
        if model is None:
            return None
        return cls(model=model, preprocessor=preprocessor)

    def predict_ratio(self, semantic_score: float, keyword_score: float, length_score: float) -> float:
        """Predict normalized marks ratio bounded to [0, 1]."""
        features = np.array([[semantic_score, keyword_score, length_score]], dtype=float)
        if self.preprocessor is not None:
            row_df = pd.DataFrame(
                [
                    {
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_score,
                        "length_score": length_score,
                    }
                ]
            )
            features = self.preprocessor.transform(row_df)
        ratio = float(self.model.predict(features)[0])
        return max(0.0, min(ratio, 1.0))

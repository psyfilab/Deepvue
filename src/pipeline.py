"""Library orchestration layer for dataframe-based inference."""

from __future__ import annotations

import pandas as pd

from .predict import KeywordClassifier


class BankKeywordPipeline:
    """Run the trained classifier over a dataframe column."""

    def __init__(self, models_dir: str = "models/") -> None:
        """Create a dataframe-oriented inference pipeline."""

        self.classifier = KeywordClassifier(models_dir=models_dir)

    def run(self, df: pd.DataFrame, keyword_column: str = "Key Word") -> pd.DataFrame:
        """Return the input dataframe with predicted level columns appended."""

        if keyword_column not in df.columns:
            raise KeyError(f"Column not found: {keyword_column}")

        keywords = df[keyword_column].fillna("").astype(str).tolist()
        predictions = self.classifier.classify_batch(keywords)
        predictions_df = pd.DataFrame(predictions)
        return pd.concat([df.reset_index(drop=True), predictions_df], axis=1)

"""Keyword text normalization utilities."""

from __future__ import annotations

import re
from typing import Iterable


class KeywordPreprocessor:
    """Normalize bank narration keywords into model-friendly text."""

    placeholder_token = "NUMTOKEN"
    _digit_pattern = re.compile(r"\d+")
    _special_pattern = re.compile(r"[^a-zA-Z\s-]")
    _space_pattern = re.compile(r"\s+")
    _placeholder_pattern = re.compile(r"\bnumtoken\b", flags=re.IGNORECASE)

    def transform(self, keyword: str) -> str:
        """Return a cleaned version of the incoming keyword."""

        text = "" if keyword is None else str(keyword)
        text = text.strip().lower()
        if not text:
            return ""

        text = self._digit_pattern.sub(f" {self.placeholder_token} ", text)
        text = self._special_pattern.sub(" ", text)
        text = self._space_pattern.sub(" ", text).strip()
        text = self._placeholder_pattern.sub(self.placeholder_token, text)
        return text

    def transform_batch(self, keywords: Iterable[str]) -> list[str]:
        """Normalize a batch of keywords."""

        return [self.transform(keyword) for keyword in keywords]

    def __getstate__(self) -> dict[str, str]:
        """Return a serializable representation of the stateless preprocessor."""

        return {}

    def __setstate__(self, state: dict[str, str]) -> None:
        """Restore the stateless preprocessor."""

        _ = state

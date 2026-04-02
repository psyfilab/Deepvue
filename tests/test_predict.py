"""Integration tests for the prompt-aligned PART_3 classifier."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.predict import KeywordClassifier
from src.preprocess import KeywordPreprocessor
from src.prompt_part3 import PromptArtifacts


def _build_dummy_bundle(models_dir: Path) -> None:
    """Create a tiny prompt bundle for tests."""

    preprocessor = KeywordPreprocessor()
    exact_lookup = {
        preprocessor.transform("arun"): {
            "Level_0": "",
            "Level_1": "",
            "Level_2": "",
            "Level_3": "p2p",
            "Level_4": "",
            "Level_5": "",
            "Level_6": "arun",
            "Level_7": "",
        },
        preprocessor.transform("zomato"): {
            "Level_0": "",
            "Level_1": "",
            "Level_2": "",
            "Level_3": "merchant",
            "Level_4": "",
            "Level_5": "dining",
            "Level_6": "zomato",
            "Level_7": "",
        },
    }
    bundle = PromptArtifacts(
        preprocessor=preprocessor,
        exact_lookup=exact_lookup,
        level3_model=None,
        level5_model=None,
        merchant_aliases={
            "zomato": {"Level_3": "merchant", "Level_5": "dining", "Level_6": "zomato"}
        },
        platform_aliases={"hdfc": "hdfc bank"},
        metadata={},
    )
    joblib.dump(bundle, models_dir / "prompt_part3_bundle.joblib")


def test_classify_returns_exactly_eight_keys(tmp_path: Path) -> None:
    """Single-keyword inference should always emit all output levels."""

    _build_dummy_bundle(tmp_path)
    classifier = KeywordClassifier(str(tmp_path))

    result = classifier.classify("arun")

    assert set(result.keys()) == {f"Level_{index}" for index in range(8)}


def test_classify_batch_matches_single_row_inference(tmp_path: Path) -> None:
    """Batch inference should agree with per-row inference."""

    _build_dummy_bundle(tmp_path)
    classifier = KeywordClassifier(str(tmp_path))
    keywords = ["arun", "zomato", "hdfc"]

    batch_results = classifier.classify_batch(keywords)
    single_results = [classifier.classify(keyword) for keyword in keywords]

    assert batch_results == single_results

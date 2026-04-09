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
        level5_keyword_cues={"apple": "pharmacy", "p2m": "digital services"},
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


def test_level5_substring_cue_applies_without_exact_match(tmp_path: Path) -> None:
    """Longer keywords should still pick up a matching Level 5 cue."""

    _build_dummy_bundle(tmp_path)
    classifier = KeywordClassifier(str(tmp_path))

    result = classifier.classify("apple med")

    assert result["Level_5"] == "pharmacy"
    assert result["Level_3"] == "merchant"


def test_classify_with_metadata_flags_weak_unseen_merchant(tmp_path: Path) -> None:
    """Weak unseen merchant guesses should be surfaced for review."""

    _build_dummy_bundle(tmp_path)
    classifier = KeywordClassifier(str(tmp_path))

    result = classifier.classify_with_metadata("urban tech systems")

    assert result["Level_3"] == "merchant"
    assert int(result["review_unknown"]) == 1
    assert "business_name_heuristic" in result["review_unknown_reason"]

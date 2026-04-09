"""Predict prompt-aligned PART_3 classifications for keywords."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.prompt_part3 import (
        PromptArtifacts,
        clean_entity,
        detect_counterparty_marker,
        detect_explicit_purpose,
        detect_level5_keyword_cue,
        detect_manual_merchant,
        detect_p2p_entity,
        detect_platform_or_bank,
        empty_result,
        format_prompt_tsv,
        is_rail_platform_only_keyword,
        looks_like_business_name,
        looks_like_person_name,
    )
    from src.rule_engine import RuleEngine
else:
    from .prompt_part3 import (
        PromptArtifacts,
        clean_entity,
        detect_counterparty_marker,
        detect_explicit_purpose,
        detect_level5_keyword_cue,
        detect_manual_merchant,
        detect_p2p_entity,
        detect_platform_or_bank,
        empty_result,
        format_prompt_tsv,
        is_rail_platform_only_keyword,
        looks_like_business_name,
        looks_like_person_name,
    )
    from .rule_engine import RuleEngine

logger = logging.getLogger(__name__)

WEAK_LEVEL3_SOURCES = frozenset({"business_name_heuristic", "p2p_entity_heuristic"})
LOW_CONFIDENCE_LEVEL3_THRESHOLD = 0.78
LOW_CONFIDENCE_LEVEL5_THRESHOLD = 0.82


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration."""

    with Path(config_path).open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    """Resolve a config path relative to the repository root."""

    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


class KeywordClassifier:
    """Prompt-aligned PART_3 keyword classifier."""

    def __init__(self, models_dir: str) -> None:
        """Load the trained prompt bundle."""

        bundle_path = Path(models_dir) / "prompt_part3_bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Missing prompt bundle in {models_dir}. "
                "Run `python src/train.py --data MASTER_SHEET.csv --output models/` first."
            )

        self.artifacts: PromptArtifacts = joblib.load(bundle_path)
        if not hasattr(self.artifacts, "level5_keyword_cues"):
            self.artifacts.level5_keyword_cues = {}
        self.rule_engine = RuleEngine()

    def _fallback_unknown(self, result: dict[str, str]) -> dict[str, str]:
        """Ensure at least one level is always filled."""

        if not any(result.values()):
            result["Level_0"] = "unknown"
        return result

    def _empty_metadata(self, raw_keyword: str, cleaned_keyword: str) -> dict[str, Any]:
        """Return a default metadata payload for one keyword."""

        return {
            "raw_keyword": raw_keyword,
            "cleaned_keyword": cleaned_keyword,
            "decision_source": "",
            "level3_source": "",
            "level3_confidence": None,
            "level5_source": "",
            "level5_confidence": None,
            "overall_confidence": 0.0,
            "review_unknown": 0,
            "review_unknown_reason": "",
        }

    def _set_level3(
        self,
        result: dict[str, str],
        metadata: dict[str, Any],
        label: str,
        source: str,
        confidence: float | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Set Level 3 and tracking metadata."""

        if force or not result["Level_3"]:
            result["Level_3"] = label
            metadata["level3_source"] = source
            if confidence is not None:
                metadata["level3_confidence"] = round(float(confidence), 4)

    def _set_level5(
        self,
        result: dict[str, str],
        metadata: dict[str, Any],
        label: str,
        source: str,
        confidence: float | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Set Level 5 and tracking metadata."""

        if force or not result["Level_5"]:
            result["Level_5"] = label
            metadata["level5_source"] = source
            if confidence is not None:
                metadata["level5_confidence"] = round(float(confidence), 4)

    def _finalize_metadata(
        self,
        result: dict[str, str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute summary confidence and review flags."""

        confidence_values = [
            value
            for value in (
                metadata.get("level3_confidence"),
                metadata.get("level5_confidence"),
            )
            if value is not None
        ]

        if metadata["decision_source"] == "exact_lookup":
            metadata["overall_confidence"] = 1.0
        elif confidence_values:
            metadata["overall_confidence"] = round(float(min(confidence_values)), 4)
        elif any(result.get(level_name) for level_name in ("Level_0", "Level_1", "Level_2", "Level_7")):
            metadata["overall_confidence"] = 1.0
        else:
            metadata["overall_confidence"] = 0.0

        review_reasons: list[str] = []
        level3_confidence = metadata.get("level3_confidence")
        level5_confidence = metadata.get("level5_confidence")

        if result["Level_0"] == "unknown":
            review_reasons.append("fallback_unknown")
        if result["Level_3"] in {"merchant", "p2p"}:
            if metadata.get("level3_source") in WEAK_LEVEL3_SOURCES:
                review_reasons.append(metadata["level3_source"])
            if (
                level3_confidence is not None
                and level3_confidence < LOW_CONFIDENCE_LEVEL3_THRESHOLD
            ):
                review_reasons.append("low_level3_confidence")
        if (
            result["Level_3"] == "merchant"
            and result["Level_5"]
            and level5_confidence is not None
            and level5_confidence < LOW_CONFIDENCE_LEVEL5_THRESHOLD
        ):
            review_reasons.append("low_level5_confidence")

        metadata["review_unknown"] = int(bool(review_reasons))
        metadata["review_unknown_reason"] = ",".join(dict.fromkeys(review_reasons))
        if not metadata["decision_source"]:
            metadata["decision_source"] = (
                metadata["level3_source"]
                or metadata["level5_source"]
                or "rules"
            )
        return metadata

    def _classify_internal(self, keyword: str) -> tuple[dict[str, str], dict[str, Any]]:
        """Classify one keyword and return debug metadata."""

        raw_keyword = "" if keyword is None else str(keyword)
        cleaned = self.artifacts.preprocessor.transform(raw_keyword)
        result = empty_result()
        metadata = self._empty_metadata(raw_keyword, cleaned)

        if not raw_keyword.strip():
            result["Level_0"] = "unknown"
            metadata["decision_source"] = "empty_keyword"
            return result, self._finalize_metadata(result, metadata)

        exact_match = self.artifacts.exact_lookup.get(cleaned)
        if exact_match:
            metadata["decision_source"] = "exact_lookup"
            metadata["level3_source"] = "exact_lookup" if exact_match.get("Level_3") else ""
            metadata["level5_source"] = "exact_lookup" if exact_match.get("Level_5") else ""
            metadata["level3_confidence"] = 1.0 if exact_match.get("Level_3") else None
            metadata["level5_confidence"] = 1.0 if exact_match.get("Level_5") else None
            return exact_match.copy(), self._finalize_metadata(exact_match.copy(), metadata)

        result.update(self.rule_engine.classify(raw_keyword))
        if any(result.get(level_name) for level_name in ("Level_0", "Level_1", "Level_2", "Level_7")):
            metadata["decision_source"] = "rules"

        if result["Level_1"] and is_rail_platform_only_keyword(raw_keyword):
            result = self._fallback_unknown(result)
            metadata["decision_source"] = "rail_platform_only"
            return result, self._finalize_metadata(result, metadata)

        manual_merchant = detect_manual_merchant(raw_keyword)
        if manual_merchant:
            result.update({key: value for key, value in manual_merchant.items() if value})
            if manual_merchant.get("Level_3"):
                self._set_level3(
                    result,
                    metadata,
                    manual_merchant["Level_3"],
                    "manual_merchant_alias",
                    0.97,
                    force=True,
                )
            if manual_merchant.get("Level_5"):
                self._set_level5(
                    result,
                    metadata,
                    manual_merchant["Level_5"],
                    "manual_merchant_alias",
                    0.97,
                    force=True,
                )
            metadata["decision_source"] = "manual_merchant_alias"

        if not result["Level_3"]:
            counterparty_marker = detect_counterparty_marker(raw_keyword)
            if counterparty_marker:
                self._set_level3(
                    result,
                    metadata,
                    counterparty_marker,
                    "counterparty_marker",
                    0.99,
                )
                metadata["decision_source"] = metadata["decision_source"] or "counterparty_marker"

        compact = cleaned.replace(" ", "")
        if not manual_merchant:
            for alias, payload in sorted(
                self.artifacts.merchant_aliases.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            ):
                if alias in compact:
                    result.update({key: value for key, value in payload.items() if value})
                    if payload.get("Level_3"):
                        self._set_level3(
                            result,
                            metadata,
                            payload["Level_3"],
                            "merchant_alias",
                            0.95,
                            force=True,
                        )
                    if payload.get("Level_5"):
                        self._set_level5(
                            result,
                            metadata,
                            payload["Level_5"],
                            "merchant_alias",
                            0.95,
                            force=True,
                        )
                    metadata["decision_source"] = metadata["decision_source"] or "merchant_alias"
                    break

        if not result["Level_3"]:
            p2p_entity = detect_p2p_entity(raw_keyword, self.artifacts.merchant_aliases)
            if p2p_entity:
                self._set_level3(
                    result,
                    metadata,
                    "p2p",
                    "p2p_entity_heuristic",
                    0.66,
                )
                result["Level_6"] = p2p_entity
                metadata["decision_source"] = metadata["decision_source"] or "p2p_entity_heuristic"

        if not result["Level_3"]:
            platform = detect_platform_or_bank(raw_keyword, self.artifacts.merchant_aliases)
            if platform:
                self._set_level3(
                    result,
                    metadata,
                    "institution",
                    "platform_alias",
                    0.93,
                )
                result["Level_4"] = platform
                metadata["decision_source"] = metadata["decision_source"] or "platform_alias"

        if not result["Level_3"] and looks_like_business_name(raw_keyword):
            self._set_level3(
                result,
                metadata,
                "merchant",
                "business_name_heuristic",
                0.58,
            )
            result["Level_6"] = result["Level_6"] or clean_entity(raw_keyword)
            metadata["decision_source"] = metadata["decision_source"] or "business_name_heuristic"

        if not result["Level_3"] and self.artifacts.level3_model is not None:
            probabilities = self.artifacts.level3_model.predict_proba([cleaned])[0]
            classes = self.artifacts.level3_model.named_steps["clf"].classes_
            best_index = int(probabilities.argmax())
            best_label = str(classes[best_index])
            best_probability = float(probabilities[best_index])
            if best_probability >= 0.72 and (
                best_label != "p2p" or looks_like_person_name(raw_keyword)
            ):
                self._set_level3(
                    result,
                    metadata,
                    best_label,
                    "level3_model",
                    best_probability,
                )
                if best_label == "p2p" and not result["Level_6"]:
                    result["Level_6"] = clean_entity(raw_keyword)
                metadata["decision_source"] = metadata["decision_source"] or "level3_model"

        if not result["Level_5"]:
            explicit_purpose = detect_explicit_purpose(raw_keyword)
            if explicit_purpose:
                self._set_level5(
                    result,
                    metadata,
                    explicit_purpose,
                    "explicit_purpose",
                    0.96,
                )
                if (
                    result["Level_3"] == "p2p"
                    and result["Level_6"] == clean_entity(raw_keyword)
                    and explicit_purpose not in {"p2p transfer", "salary"}
                ):
                    self._set_level3(
                        result,
                        metadata,
                        "merchant",
                        "purpose_override",
                        min(metadata.get("level3_confidence") or 0.58, 0.72),
                        force=True,
                    )
                metadata["decision_source"] = metadata["decision_source"] or "explicit_purpose"

        if not result["Level_5"]:
            cue_purpose = detect_level5_keyword_cue(
                raw_keyword,
                self.artifacts.level5_keyword_cues,
            )
            if cue_purpose:
                self._set_level5(
                    result,
                    metadata,
                    cue_purpose,
                    "level5_keyword_cue",
                    0.83,
                )
                if (
                    result["Level_3"] == "p2p"
                    and result["Level_6"] == clean_entity(raw_keyword)
                    and cue_purpose not in {"p2p transfer", "salary"}
                ):
                    self._set_level3(
                        result,
                        metadata,
                        "merchant",
                        "purpose_override",
                        min(metadata.get("level3_confidence") or 0.58, 0.74),
                        force=True,
                    )
                metadata["decision_source"] = metadata["decision_source"] or "level5_keyword_cue"

        if (
            not result["Level_5"]
            and self.artifacts.level5_model is not None
            and result["Level_3"] in {"merchant", "institution"}
        ):
            probabilities = self.artifacts.level5_model.predict_proba([cleaned])[0]
            classes = self.artifacts.level5_model.named_steps["clf"].classes_
            best_index = int(probabilities.argmax())
            best_label = str(classes[best_index])
            best_probability = float(probabilities[best_index])
            if best_probability >= 0.80:
                self._set_level5(
                    result,
                    metadata,
                    best_label,
                    "level5_model",
                    best_probability,
                )
                metadata["decision_source"] = metadata["decision_source"] or "level5_model"

        if result["Level_3"] == "p2p":
            result["Level_4"] = ""
            if "transfer" not in raw_keyword.lower() and result["Level_5"] == "p2p transfer":
                result["Level_5"] = ""

        if result["Level_3"] == "institution" and result["Level_4"]:
            result["Level_6"] = ""

        result["Level_6"] = clean_entity(result["Level_6"], result["Level_4"])
        result = self._fallback_unknown(result)
        if result["Level_0"] == "unknown" and not metadata["decision_source"]:
            metadata["decision_source"] = "fallback_unknown"
        return result, self._finalize_metadata(result, metadata)

    def classify(self, keyword: str) -> dict[str, str]:
        """Classify a single keyword."""

        result, _ = self._classify_internal(keyword)
        return result

    def classify_with_metadata(self, keyword: str) -> dict[str, Any]:
        """Classify one keyword and return labels plus confidence metadata."""

        result, metadata = self._classify_internal(keyword)
        return {**result, **metadata}

    def classify_batch(self, keywords: list[str]) -> list[dict[str, str]]:
        """Classify a batch of keywords."""

        return [self.classify(keyword) for keyword in keywords]

    def classify_batch_with_metadata(self, keywords: list[str]) -> list[dict[str, Any]]:
        """Classify a batch of keywords with review metadata."""

        return [self.classify_with_metadata(keyword) for keyword in keywords]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Predict PART_3 prompt classifications.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--keyword", help="Single keyword to classify")
    parser.add_argument("--input-file", help="One keyword per line")
    parser.add_argument("--models", help="Directory containing trained bundle")
    parser.add_argument("--output-file", help="Optional TSV output file")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    if not args.keyword and not args.input_file:
        raise ValueError("Provide either --keyword or --input-file")

    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(args.config)
    models_dir = resolve_path(repo_root, args.models or config["output"]["models_dir"])
    classifier = KeywordClassifier(str(models_dir))

    if args.keyword:
        keywords = [args.keyword]
    else:
        keywords = Path(args.input_file).read_text(encoding="utf-8").splitlines()

    predictions = classifier.classify_batch(keywords)
    tsv_output = format_prompt_tsv(keywords, predictions)

    if args.output_file:
        Path(args.output_file).write_text(tsv_output, encoding="utf-8")

    print(tsv_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

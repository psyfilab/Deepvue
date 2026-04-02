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
        detect_explicit_purpose,
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
        detect_explicit_purpose,
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
        self.rule_engine = RuleEngine()

    def _fallback_unknown(self, result: dict[str, str]) -> dict[str, str]:
        """Ensure at least one level is always filled."""

        if not any(result.values()):
            result["Level_0"] = "unknown"
        return result

    def classify(self, keyword: str) -> dict[str, str]:
        """Classify a single keyword."""

        raw_keyword = "" if keyword is None else str(keyword)
        cleaned = self.artifacts.preprocessor.transform(raw_keyword)
        result = empty_result()

        if not raw_keyword.strip():
            result["Level_0"] = "unknown"
            return result

        exact_match = self.artifacts.exact_lookup.get(cleaned)
        if exact_match:
            return exact_match.copy()

        result.update(self.rule_engine.classify(raw_keyword))

        if result["Level_1"] and is_rail_platform_only_keyword(raw_keyword):
            return self._fallback_unknown(result)

        manual_merchant = detect_manual_merchant(raw_keyword)
        if manual_merchant:
            result.update({key: value for key, value in manual_merchant.items() if value})

        compact = cleaned.replace(" ", "")
        if not manual_merchant:
            for alias, payload in sorted(
                self.artifacts.merchant_aliases.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            ):
                if alias in compact:
                    result.update({key: value for key, value in payload.items() if value})
                    break

        if not result["Level_3"]:
            p2p_entity = detect_p2p_entity(raw_keyword, self.artifacts.merchant_aliases)
            if p2p_entity:
                result["Level_3"] = "p2p"
                result["Level_6"] = p2p_entity

        if not result["Level_3"]:
            platform = detect_platform_or_bank(raw_keyword, self.artifacts.merchant_aliases)
            if platform:
                result["Level_3"] = "institution"
                result["Level_4"] = platform

        if not result["Level_3"] and looks_like_business_name(raw_keyword):
            result["Level_3"] = "merchant"
            result["Level_6"] = result["Level_6"] or clean_entity(raw_keyword)

        if not result["Level_3"] and self.artifacts.level3_model is not None:
            probabilities = self.artifacts.level3_model.predict_proba([cleaned])[0]
            classes = self.artifacts.level3_model.named_steps["clf"].classes_
            best_index = int(probabilities.argmax())
            best_label = str(classes[best_index])
            best_probability = float(probabilities[best_index])
            if best_probability >= 0.72 and (
                best_label != "p2p" or looks_like_person_name(raw_keyword)
            ):
                result["Level_3"] = best_label
                if best_label == "p2p" and not result["Level_6"]:
                    result["Level_6"] = clean_entity(raw_keyword)

        if not result["Level_5"]:
            explicit_purpose = detect_explicit_purpose(raw_keyword)
            if explicit_purpose:
                result["Level_5"] = explicit_purpose

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
                result["Level_5"] = best_label

        if result["Level_3"] == "p2p":
            result["Level_4"] = ""
            if "transfer" not in raw_keyword.lower() and result["Level_5"] == "p2p transfer":
                result["Level_5"] = ""

        if result["Level_3"] == "institution" and result["Level_4"]:
            result["Level_6"] = ""

        result["Level_6"] = clean_entity(result["Level_6"], result["Level_4"])
        return self._fallback_unknown(result)

    def classify_batch(self, keywords: list[str]) -> list[dict[str, str]]:
        """Classify a batch of keywords."""

        return [self.classify(keyword) for keyword in keywords]


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

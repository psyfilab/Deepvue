"""Train the prompt-aligned PART_3 keyword classifier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.data_loader import load_master
    from src.features import build_feature_pipeline
    from src.preprocess import KeywordPreprocessor
    from src.prompt_part3 import (
        MANUAL_PLATFORM_ALIASES,
        PromptArtifacts,
        build_exact_lookup,
        build_initial_prompt_dataset,
        build_merchant_aliases,
        rebuild_prompt_dataset,
    )
else:
    from .data_loader import load_master
    from .features import build_feature_pipeline
    from .preprocess import KeywordPreprocessor
    from .prompt_part3 import (
        MANUAL_PLATFORM_ALIASES,
        PromptArtifacts,
        build_exact_lookup,
        build_initial_prompt_dataset,
        build_merchant_aliases,
        rebuild_prompt_dataset,
    )

logger = logging.getLogger(__name__)

PROCESSED_KEYWORD_COLUMN = "_processed_keyword"


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration."""

    with Path(config_path).open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    """Resolve a config path relative to the repository root."""

    path = Path(raw_path)
    return path if path.is_absolute() else base_dir / path


def build_classifier(C: float = 1.0) -> Pipeline:
    """Build a logistic-regression text classifier."""

    return Pipeline(
        steps=[
            ("features", build_feature_pipeline()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def train_prompt_models(train_df: pd.DataFrame) -> tuple[Pipeline, Pipeline]:
    """Train Level 3 and Level 5 prompt models."""

    level3_subset = train_df[train_df["Level_3"] != ""].copy()
    level5_subset = train_df[train_df["Level_5"] != ""].copy()

    level3_model = build_classifier(C=1.0)
    level3_model.fit(
        level3_subset[PROCESSED_KEYWORD_COLUMN].tolist(),
        level3_subset["Level_3"].tolist(),
        clf__sample_weight=level3_subset["Count"].tolist(),
    )

    level5_model = build_classifier(C=1.5)
    level5_model.fit(
        level5_subset[PROCESSED_KEYWORD_COLUMN].tolist(),
        level5_subset["Level_5"].tolist(),
        clf__sample_weight=level5_subset["Count"].tolist(),
    )

    return level3_model, level5_model


def apply_bundle_prediction(
    keyword: str,
    artifacts: PromptArtifacts,
) -> dict[str, str]:
    """Apply the trained prompt bundle to one keyword."""

    from src.prompt_part3 import (
        clean_entity,
        detect_explicit_purpose,
        detect_manual_merchant,
        detect_p2p_entity,
        detect_platform_or_bank,
        empty_result,
        is_rail_platform_only_keyword,
        looks_like_business_name,
        looks_like_person_name,
    )

    result = empty_result()
    cleaned = artifacts.preprocessor.transform(keyword)
    if not cleaned:
        result["Level_0"] = "unknown"
        return result

    exact_match = artifacts.exact_lookup.get(cleaned)
    if exact_match:
        return exact_match.copy()

    from src.rule_engine import RuleEngine

    result.update(RuleEngine().classify(keyword))

    if result["Level_1"] and is_rail_platform_only_keyword(keyword):
        return result

    merchant_hit = detect_manual_merchant(keyword)
    for alias, payload in artifacts.merchant_aliases.items():
        if alias in cleaned.replace(" ", ""):
            merchant_hit = payload.copy()
            break

    if merchant_hit:
        result.update({key: value for key, value in merchant_hit.items() if value})

    if not result["Level_3"]:
        p2p_entity = detect_p2p_entity(keyword, artifacts.merchant_aliases)
        if p2p_entity:
            result["Level_3"] = "p2p"
            result["Level_6"] = p2p_entity

    if not result["Level_3"]:
        platform = detect_platform_or_bank(keyword, artifacts.merchant_aliases)
        if platform:
            result["Level_3"] = "institution"
            result["Level_4"] = platform

    if not result["Level_3"] and looks_like_business_name(keyword):
        result["Level_3"] = "merchant"
        result["Level_6"] = result["Level_6"] or clean_entity(keyword)

    if not result["Level_3"] and artifacts.level3_model is not None:
        probabilities = artifacts.level3_model.predict_proba([cleaned])[0]
        classes = artifacts.level3_model.named_steps["clf"].classes_
        best_index = int(probabilities.argmax())
        best_label = str(classes[best_index])
        best_probability = float(probabilities[best_index])
        if best_probability >= 0.72 and (
            best_label != "p2p" or looks_like_person_name(keyword)
        ):
            result["Level_3"] = best_label
            if best_label == "p2p" and not result["Level_6"]:
                result["Level_6"] = clean_entity(keyword)

    if not result["Level_5"]:
        explicit_purpose = detect_explicit_purpose(keyword)
        if explicit_purpose:
            result["Level_5"] = explicit_purpose

    if (
        not result["Level_5"]
        and artifacts.level5_model is not None
        and result["Level_3"] in {"merchant", "institution"}
    ):
        probabilities = artifacts.level5_model.predict_proba([cleaned])[0]
        classes = artifacts.level5_model.named_steps["clf"].classes_
        best_index = int(probabilities.argmax())
        best_label = str(classes[best_index])
        best_probability = float(probabilities[best_index])
        if best_probability >= 0.80:
            result["Level_5"] = best_label

    if result["Level_3"] == "p2p":
        result["Level_4"] = ""
        if "transfer" not in keyword.lower() and result["Level_5"] == "p2p transfer":
            result["Level_5"] = ""

    if result["Level_3"] == "institution" and result["Level_4"]:
        result["Level_6"] = ""

    result["Level_6"] = clean_entity(result["Level_6"], result["Level_4"])

    if not any(result.values()):
        result["Level_0"] = "unknown"

    return result


def evaluate_holdout(
    test_df: pd.DataFrame,
    artifacts: PromptArtifacts,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate the prompt bundle on the holdout split."""

    predictions = [apply_bundle_prediction(keyword, artifacts) for keyword in test_df["Key Word"]]
    predictions_df = pd.DataFrame(predictions)

    comparison = pd.concat(
        [
            test_df[["Key Word", "Count", "Level_0", "Level_1", "Level_2", "Level_3", "Level_4", "Level_5", "Level_6", "Level_7"]]
            .reset_index(drop=True),
            predictions_df.add_prefix("Pred_"),
        ],
        axis=1,
    )
    comparison.to_csv(output_path, sep="\t", index=False)

    weights = test_df["Count"].tolist()
    exact_match = [
        int(
            all(
                str(test_df.iloc[index][column]) == str(predictions_df.iloc[index][column])
                for column in ["Level_0", "Level_1", "Level_2", "Level_3", "Level_4", "Level_5", "Level_6", "Level_7"]
            )
        )
        for index in range(len(test_df))
    ]

    metrics: dict[str, Any] = {
        "num_test_rows": int(len(test_df)),
        "weighted_exact_match": float(sum(match * weight for match, weight in zip(exact_match, weights)) / max(sum(weights), 1)),
        "per_level_accuracy": {},
        "level3_macro_f1": None,
        "level5_macro_f1": None,
    }

    for column in ["Level_0", "Level_1", "Level_2", "Level_3", "Level_4", "Level_5", "Level_6", "Level_7"]:
        metrics["per_level_accuracy"][column] = float(
            accuracy_score(test_df[column].tolist(), predictions_df[column].tolist(), sample_weight=weights)
        )

    level3_mask = test_df["Level_3"] != ""
    if level3_mask.any():
        metrics["level3_macro_f1"] = float(
            f1_score(
                test_df.loc[level3_mask, "Level_3"],
                predictions_df.loc[level3_mask, "Level_3"],
                average="macro",
                zero_division=0,
            )
        )
        metrics["level3_report"] = classification_report(
            test_df.loc[level3_mask, "Level_3"],
            predictions_df.loc[level3_mask, "Level_3"],
            zero_division=0,
        )

    level5_mask = test_df["Level_5"] != ""
    if level5_mask.any():
        metrics["level5_macro_f1"] = float(
            f1_score(
                test_df.loc[level5_mask, "Level_5"],
                predictions_df.loc[level5_mask, "Level_5"],
                average="macro",
                zero_division=0,
            )
        )
        metrics["level5_report"] = classification_report(
            test_df.loc[level5_mask, "Level_5"],
            predictions_df.loc[level5_mask, "Level_5"],
            zero_division=0,
        )

    return metrics


def build_raw_part3_sample(raw_data_dir: Path, output_path: Path, limit: int = 200) -> None:
    """Extract unseen PART_3 keywords from raw Excel files for smoke testing."""

    import re

    rows: list[str] = []
    for path in sorted(raw_data_dir.glob("*.xlsx")):
        try:
            df = pd.read_excel(path, sheet_name="Bank Statement", header=4, engine="openpyxl")
        except Exception:
            continue

        if "Particulars" not in df.columns:
            continue

        particulars = df["Particulars"].fillna("").astype(str).str.strip()
        particulars = particulars[(particulars != "") & (particulars.str.lower() != "nan")]

        for narration in particulars:
            text = narration.upper()
            text = re.sub(r"\bTO\s+TRANSFER\s*-", "TO TRANSFER/", text)
            text = re.sub(r"\bBY\s+TRANSFER\s*-", "BY TRANSFER/", text)
            text = re.sub(r"\d+", "", text)
            parts = [part.strip() for part in re.split(r"[/\-*]", text) if part.strip()]
            if len(parts) > 3:
                rows.append(parts[3])

    sample = pd.Series(rows).drop_duplicates().head(limit)
    output_path.write_text("\n".join(sample.tolist()), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train the PART_3 prompt classifier.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", help="Path to MASTER_SHEET.csv")
    parser.add_argument("--output", help="Directory to store trained models")
    parser.add_argument("--reports-dir", help="Directory to store reports")
    parser.add_argument("--test-size", type=float, help="Holdout size override")
    parser.add_argument("--random-state", type=int, help="Random state override")
    parser.add_argument("--raw-data-dir", help="Optional raw-data directory for sample extraction")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(args.config)

    data_path = resolve_path(repo_root, args.data or config["data"]["master_sheet_path"])
    models_dir = resolve_path(repo_root, args.output or config["output"]["models_dir"])
    reports_dir = resolve_path(repo_root, args.reports_dir or config["output"]["reports_dir"])
    test_size = float(args.test_size or config["data"]["test_size"])
    random_state = int(args.random_state or config["data"]["random_state"])
    raw_data_dir = resolve_path(repo_root, args.raw_data_dir or "Raw Data")

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    master_df = load_master(str(data_path))
    prompt_df = build_initial_prompt_dataset(master_df)
    merchant_aliases = build_merchant_aliases(prompt_df)
    prompt_df = rebuild_prompt_dataset(prompt_df, merchant_aliases)

    preprocessor = KeywordPreprocessor()
    prompt_df[PROCESSED_KEYWORD_COLUMN] = preprocessor.transform_batch(prompt_df["Key Word"].tolist())

    train_df, test_df = train_test_split(
        prompt_df,
        test_size=test_size,
        random_state=random_state,
        stratify=prompt_df["Level_3"].replace("", "unknown"),
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    level3_model, level5_model = train_prompt_models(train_df)
    artifacts = PromptArtifacts(
        preprocessor=preprocessor,
        exact_lookup=build_exact_lookup(train_df, preprocessor),
        level3_model=level3_model,
        level5_model=level5_model,
        merchant_aliases=merchant_aliases,
        platform_aliases={alias: name for name, aliases in MANUAL_PLATFORM_ALIASES.items() for alias in aliases},
        metadata={},
    )

    holdout_path = reports_dir / "part3_holdout_predictions.tsv"
    metrics = evaluate_holdout(test_df, artifacts, holdout_path)

    artifacts.metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "scope": "PART_3",
        "num_train_rows": int(len(train_df)),
        "num_test_rows": int(len(test_df)),
        "weighted_exact_match": metrics["weighted_exact_match"],
        "per_level_accuracy": metrics["per_level_accuracy"],
        "level3_macro_f1": metrics["level3_macro_f1"],
        "level5_macro_f1": metrics["level5_macro_f1"],
    }

    joblib.dump(artifacts, models_dir / "prompt_part3_bundle.joblib")
    with (models_dir / "metadata.json").open("w", encoding="utf-8") as file_handle:
        json.dump(artifacts.metadata, file_handle, indent=2)
    prompt_df.drop(columns=[PROCESSED_KEYWORD_COLUMN]).to_csv(
        reports_dir / "part3_prompt_dataset.csv",
        index=False,
    )
    with (reports_dir / "part3_metrics.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    if raw_data_dir.exists():
        build_raw_part3_sample(raw_data_dir, repo_root / "data" / "sample_part3_keywords.txt")

    print(f"Trained PART_3 prompt classifier on {len(train_df)} rows")
    print(f"Weighted exact holdout match: {metrics['weighted_exact_match']:.4f}")
    print(f"Level_3 macro F1: {metrics['level3_macro_f1']}")
    print(f"Level_5 macro F1: {metrics['level5_macro_f1']}")
    print(f"Bundle saved to: {models_dir / 'prompt_part3_bundle.joblib'}")
    print(f"Holdout predictions saved to: {holdout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

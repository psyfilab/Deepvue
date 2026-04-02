# Bank Keyword Classifier

This repo now trains a prompt-aligned `Part_3` keyword classifier from `MASTER_SHEET.csv`. It is local-first, uses no external inference API, and is designed to behave conservatively: exact lookup first, rules second, ML fallback last.

## Install

```bash
python -m pip install -r requirements.txt
```

## Data

Expected columns in `MASTER_SHEET.csv`:

```text
Key Word, Source, Count, Level 0, Level 1, Level 2, Level 3, Level 4, Level 5, Level 6, Level 7
```

The trainer currently learns from `Source == PART_3`, because that is the prompt scope.

Place your real `MASTER_SHEET.csv` in the repo root:

```text
Deepvue/
  MASTER_SHEET.csv
```

The repo already includes safe sample files in `data/` for testing the flow.

## Train

```bash
make train
```

Or:

```bash
python src/train.py --data MASTER_SHEET.csv --output models/
```

Main outputs:

- `models/prompt_part3_bundle.joblib`
- `models/metadata.json`
- `reports/part3_metrics.json`
- `reports/part3_holdout_predictions.tsv`
- `reports/part3_prompt_dataset.csv`

## Predict

Single keyword:

```bash
python src/predict.py --keyword "hdfc" --models models/
```

Batch file:

```bash
python src/predict.py --input-file data/sample_part3_keywords.txt --models models/ --output-file reports/sample_part3_predictions.tsv
```

Output format matches the prompt style: one TSV block with `Key Word` plus `Level 0` to `Level 7`.

## Where To Put Input

For training:

- Put your full labeled file at `MASTER_SHEET.csv` in the repo root.

For batch keyword prediction:

- Create a plain text file anywhere in the repo, for example `data/my_keywords.txt`
- Put one `Part_3` keyword per line

Example:

```text
arun
hdfc
phonep e
zomat o
```

Then run:

```bash
python src/predict.py --input-file data/my_keywords.txt --models models/ --output-file reports/my_predictions.tsv
```

For full bank-statement testing:

- Put a statement CSV like `data/sample_bank_statement.csv` in `data/`
- Run `make statement-sample` or call `src/run_statement_pipeline.py` directly

For full folder processing:

- Put every new bank statement file in `Raw Data/`
- Keep the same `Bank Statement` sheet and header layout as `Raw Data/1.xlsx`
- Run `make statement-folder`

## Where To See Output

- Training metrics: `reports/part3_metrics.json`
- Row-by-row holdout results: `reports/part3_holdout_predictions.tsv`
- Batch keyword output: `reports/my_predictions.tsv`
- Sample keyword output: `reports/sample_part3_predictions.tsv`
- Statement pipeline workbook: `reports/sample_bank_statement_output.xlsx`
- Combined folder workbook: `reports/all_statements_classified.xlsx`

## Samples

Included samples:

- `data/sample_part3_keywords.txt`
- `data/sample_bank_statement.csv`

Run sample keyword prediction:

```bash
make predict-sample
```

Run the full sample statement flow:

```bash
make statement-sample
```

Run all files in `Raw Data/` at once:

```bash
make statement-folder
```

Or directly:

```bash
python src/run_statement_folder.py --input-dir "Raw Data" --models models/ --master MASTER_SHEET.csv --output reports/all_statements_classified.xlsx
```

## How It Works

- Rules handle `Level_0`, `Level_1`, `Level_2`, and `Level_7`.
- Exact lookup handles known validated `Part_3` keywords.
- Alias dictionaries handle explicit banks, platforms, and known merchants.
- A conservative logistic model backs off to `Level_3` and `Level_5` only when confidence is high.
- `Level_6` is extracted, not generatively invented.

## What Improves It

More raw statements are useful for distribution checks, but they do not improve supervision by themselves. The biggest improvement now would be a smaller audited `Part_3` validation file where labels are corrected to the prompt logic, especially for:

- person handles vs bank labels
- platform names vs merchant names
- canonical `Level_5` purpose mapping

# Deepvue Bank Statement Intelligence MVP

This repository is the working MVP for a local-first bank statement intelligence system built for [PSYFI](https://www.psyfilab.com/) and authored by Shreenand Bhattad. It classifies bank narration fragments into a structured taxonomy, enriches transactions with consumer-spend flags, detects salary credits, creates review sheets for unknown keywords, and generates statement-level analytics and plots. The system is designed to run fully offline on local files, with no external inference API and no cloud dependency at prediction time.

## Project Purpose

The core problem this MVP solves is that raw bank statement narrations are messy, inconsistent, abbreviated, and difficult to use directly for downstream analytics. A single narration may include rail information, direction, merchant markers, bank names, person names, or category hints all at once. This project converts those noisy strings into a structured, auditable output that can support classification, review operations, micro-segmentation, and behavior analytics.
The project is being maintained and improved by the PSYFI Labs team.

At a high level, the system does four things:

1. Splits each narration into reusable keyword parts.
2. Classifies those parts and rolls them up into row-level structured labels.
3. Flags spending and savings behavior for downstream analysis.
4. Produces analyst-facing Excel outputs and plots for review.

## What The Classifier Is Doing

This is not a single black-box model. It is a layered classification pipeline with deterministic rules, exact codebook lookup, heuristics, and machine learning fallback. The order matters.

### 1. Statement Splitting

Every bank narration is first split into stable pieces such as:

```text
UPI | P2M | APPLE MED | HDFC BANK | UPI MAND | P2V
```

This matters because a single narration often contains multiple signals:

- transaction rail: `UPI`
- counterparty marker: `P2M`, `P2A`, `P2V`
- entity name: `APPLE MED`
- platform / bank: `HDFC BANK`

The splitter preserves important transaction markers like `P2M`, `P2A`, `P2P`, `P2V` instead of accidentally normalizing them away.

Main file:

- [statement_splitter.py](c:/Users/DELL/OneDrive/Desktop/PSYFI/Deepvue/src/statement_splitter.py)

### 2. Keyword-Level Classification

Each part is classified with a hybrid keyword classifier trained from `MASTER_SHEET.csv`. The system follows this order:

1. Exact lookup from the validated master sheet.
2. Rule engine for deterministic levels.
3. Merchant / bank / platform alias detection.
4. Heuristics for likely person names or business names.
5. ML fallback for selected levels when confidence is high enough.

This means if a keyword already exists in the codebook, the classifier prefers the known answer. It only falls back to model behavior when it has to.

Main files:

- [predict.py](c:/Users/DELL/OneDrive/Desktop/PSYFI/Deepvue/src/predict.py)
- [prompt_part3.py](c:/Users/DELL/OneDrive/Desktop/PSYFI/Deepvue/src/prompt_part3.py)
- [rule_engine.py](c:/Users/DELL/OneDrive/Desktop/PSYFI/Deepvue/src/rule_engine.py)

### 3. Rule-Driven Levels

The deterministic levels are handled by rules instead of ML because they are explicit in the text:

- `Level_0`: direction like `debit`, `credit`
- `Level_1`: rail like `upi`, `imps`, `neft`, `rtgs`, `cash`, `cheque`, `nach`
- `Level_2`: initiation like `auto-debit`, `standing instruction`
- `Level_7`: recurring indicators like `EMI`, `SI`, `monthly`

At the statement-row level, `Level_0` is further strengthened by directly reading the actual `Debit` and `Credit` columns from the input statement.

### 4. ML Fallback

The ML part is intentionally narrow and conservative.

Current trained text models:

- `Level_3` model:
  `TF-IDF word ngrams + TF-IDF char ngrams -> LogisticRegression`
- `Level_5` model:
  `TF-IDF word ngrams + TF-IDF char ngrams -> LogisticRegression`

Why this setup:

- bank keywords are short and noisy
- character n-grams help on abbreviations and truncated merchants
- logistic regression is fast, local, and interpretable enough for thresholded use

The ML models are only used when:

- the keyword is not already known from the codebook
- rules and aliases do not fully resolve it
- model confidence passes the threshold

### 5. Structured Statement Overrides

Some row types need structure-aware overrides, especially UPI rows:

- `P2M` and `P2V` push rows toward merchant logic
- `P2A` and `P2P` push rows toward person-to-person logic
- entity extraction is tied to the correct split part
- bank/platform is inherited from the later parts when available

This prevents the system from treating the full narration as one flat string and improves row-level accuracy substantially.

Main file:

- [statement_classifier.py](c:/Users/DELL/OneDrive/Desktop/PSYFI/Deepvue/src/statement_classifier.py)

### 6. Salary Detection

Salary is not treated as just another merchant label. It is normalized into a stable path:

- `Level_0 = credit`
- `Level_3 = institution`
- `Level_5 = salary`
- `First Level Classification = Salary`
- `Second Level Classification = ""`

Salary detection uses:

- explicit salary keywords
- classification hints
- recurring high-value credits across months

The output also includes a `Salary_Summary` sheet and salary-window analytics for post-credit behavior analysis.

### 7. Micro-Segmentation

After classification, the row is enriched with behavioral flags like:

- `exp_food`
- `exp_micro`
- `exp_luxury`
- `exp_ecomm`
- `exp_entertainment`
- `exp_quick_comm`
- `exp_sub`
- `exp_emi`
- `exp_atm`
- `exp_creditcard`
- `exp_saving`
- `exp_investment`

These feed:

- `MicroSegment`
- `present_bias_score`
- savings vs expenditure ratio windows after salary credit

Main file:

- [microsegment.py](c:/Users/DELL/OneDrive/Desktop/PSYFI/Deepvue/src/microsegment.py)

### 8. Unknown Keyword Review

The batch pipeline also produces a clean `Unknown_Keywords` review tab containing only:

- `Key Word`
- `Count`
- `Source_Files`
- `Review_Status`
- `Decision_Status`

These are unseen keywords that were not found in `MASTER_SHEET.csv` and were classified with weak confidence. This is the key human-in-the-loop feedback loop for improving the codebook over time.

## MVP Workflow

The intended MVP workflow is:

1. Maintain `MASTER_SHEET.csv` as the validated codebook.
2. Train the local classifier from that codebook.
3. Drop new bank statement files into `Raw Data/`.
4. Run the folder pipeline.
5. Review:
   - classified transactions
   - salary summaries
   - salary window summaries
   - unknown keyword sheet
   - per-file plots
6. Add approved new keywords back into `MASTER_SHEET.csv`.
7. Retrain and rerun.

This is a human-in-the-loop intelligence workflow, not just a one-shot classifier.

## Repository Structure

```text
Deepvue/
  MASTER_SHEET.csv
  Raw Data/
  data/
  models/
  reports/
  src/
    data_loader.py
    features.py
    microsegment.py
    pipeline.py
    predict.py
    preprocess.py
    prompt_part3.py
    rule_engine.py
    run_statement_folder.py
    run_statement_pipeline.py
    salary_timeseries.py
    statement_classifier.py
    statement_splitter.py
    train.py
  tests/
  README.md
  requirements.txt
  Makefile
```

## Data Requirements

### MASTER_SHEET Format

Expected columns:

```text
Key Word, Source, Count, Level 0, Level 1, Level 2, Level 3, Level 4, Level 5, Level 6, Level 7
```

Current training scope:

- `Source == PART_3`

### Bank Statement Format

The folder and single-file runners expect the same structure as your existing bank files such as `Raw Data/1.xlsx`:

- sheet name: `Bank Statement`
- header row index: `4`
- columns including at least:
  - `Date`
  - `Particulars`
  - `Debit`
  - `Credit`
  - `Balance`

## Installation

```bash
python -m pip install -r requirements.txt
```

## Training

Train from the local master sheet:

```bash
python src/train.py --data MASTER_SHEET.csv --output models/ --reports-dir reports/
```

Main training outputs:

- `models/prompt_part3_bundle.joblib`
- `models/metadata.json`
- `reports/part3_metrics.json`
- `reports/part3_holdout_predictions.tsv`
- `reports/part3_prompt_dataset.csv`

## Prediction Modes

### Single Keyword

```bash
python src/predict.py --keyword "APPLE MED" --models models/
```

### Keyword Batch

```bash
python src/predict.py --input-file data/sample_part3_keywords.txt --models models/ --output-file reports/sample_part3_predictions.tsv
```

### Single Statement File

```bash
python src/run_statement_pipeline.py --input "Raw Data\\1.xlsx" --models models/ --master MASTER_SHEET.csv --window-days 10 --output reports/1_salary_timeseries_v3.xlsx
```

### Entire Raw Data Folder

```bash
python src/run_statement_folder.py --input-dir "Raw Data" --models models/ --master MASTER_SHEET.csv --window-days 10 --output reports/all_statements_salary_timeseries_v4.xlsx
```

python src/run_statement_folder.py --input-dir "Raw Data" --models models/ --master MASTER_SHEET.csv --output reports/all_statements_classified.xlsx

## Output Artifacts

### Main Workbook Sheets

Folder output workbook includes:

- `Classified_All`
- `Run_Summary`
- `Keyword_Review`
- `Unknown_Keywords`
- `Salary_Summary`
- `Salary_Window_Summary`
- `Salary_Window_Txns`

Single-file workbook includes:

- `Split`
- `Unique_Keywords`
- `Classified`
- `Salary_Summary`
- `Salary_Window_Summary`
- `Salary_Window_Txns`

### Plot Outputs

Each run also creates a dedicated plot folder beside the workbook, with one improved timeline PNG per source file. The chart includes:

- salary credit markers
- daily savings bars
- daily expenditure bars
- highlighted salary-active windows
- cumulative savings line
- cumulative expenditure line
- cumulative savings/expenditure ratio line

## Current Behavioral Ratio

The post-salary analytics now use:

```text
savings / expenditure
```

Where:

```text
savings = exp_saving + exp_investment
```

and

```text
expenditure = exp_food + exp_micro + exp_luxury + exp_ecomm +
              exp_entertainment + exp_quick_comm + exp_sub +
              exp_emi + exp_atm + exp_creditcard
```

## What Makes This MVP Strong

- Fully local inference
- Conservative classification behavior
- Explicit review loop for unknowns
- Statement-level structured overrides
- Supports both keyword-only and full statement workflows
- Produces analyst-friendly Excel outputs
- Produces visual plots for per-file behavioral analysis

## Known Limitations

- Salary detection can still over-trigger on some repeated high-value credits
- Unknown keyword review is intentionally conservative and may over-flag borderline cases
- `Level_3` remains the hardest level in noisy, unseen cases
- quality is still bottlenecked by the quality of labels in `MASTER_SHEET.csv`

## Best Next Improvements

1. Tighten salary false positives with stronger salary-entity rules.
2. Improve merchant vs person discrimination for unseen short names.
3. Add a faster keyword approval pipeline from `Unknown_Keywords` back into `MASTER_SHEET.csv`.
4. Add confidence calibration reports for `Level_3` and `Level_5`.
5. Improve versioned output folder handling for reports and plots.

## Attribution

This project is for [PSYFI](https://www.psyfilab.com/).

Authored by Shreenand Bhattad.

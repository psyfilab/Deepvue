install:
	python -m pip install -r requirements.txt

train:
	python src/train.py --data MASTER_SHEET.csv --output models/

predict:
	python src/predict.py --keyword "$(KEYWORD)" --models models/

predict-sample:
	python src/predict.py --input-file data/sample_part3_keywords.txt --models models/ --output-file reports/sample_part3_predictions.tsv

statement-sample:
	python src/run_statement_pipeline.py --input data/sample_bank_statement.csv --models models/ --master MASTER_SHEET.csv --output reports/sample_bank_statement_output.xlsx

test:
	python -m pytest tests/ -v

clean:
	python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ['models', 'reports']]"

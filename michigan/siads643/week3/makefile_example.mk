artifacts/model.pkl: permits_train.py artifacts/clean.csv
	python permits_train.py artifacts/clean.csv artifacts/model.pkl

artifacts/clean.csv: permits_clean.py artifacts/combined.csv
	python permits_clean.py artifacts/combined.csv artifacts/clean.csv

artifacts/combined.csv: permits_combine.py input_data/
	python permits_combine.py input_data/ artifacts/combined.csv

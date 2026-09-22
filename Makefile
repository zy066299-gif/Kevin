.PHONY: install data test notebook all

install:
	python -m pip install -r requirements.txt
	python -m pip install -e . --no-deps

data:
	python run_backtest.py --refresh-data

test:
	pytest

notebook:
	python -m jupyter nbconvert --execute --to notebook --inplace notebooks/phase1_backtesting.ipynb
	python scripts/export_notebook_html.py

all: data test notebook

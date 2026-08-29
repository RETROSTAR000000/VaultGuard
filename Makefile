# VaultGuard — cross-platform convenience commands
# Usage (macOS / Linux):  make setup  |  make train  |  make app
# Usage (Windows):        use run.bat instead

.PHONY: setup train importance app

setup:
	pip install -e .

train:
	python -m src.train

importance:
	python -m src.feature_importance

app:
	streamlit run app.py
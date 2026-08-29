# VaultGuard — cross-platform convenience commands (delegates to uv)
# Usage (macOS / Linux):  make setup  |  make train  |  make app
# Usage (Windows):        run.bat setup  |  run.bat train  |  run.bat app
#
# Requires: uv  (https://docs.astral.sh/uv/)
# Install:  curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: setup train importance app notebook clean

setup:
	uv sync

train:
	uv run python -m src.train

importance:
	uv run python -m src.feature_importance

app:
	uv run streamlit run app.py

notebook:
	uv run jupyter notebook notebooks/

clean:
	rm -rf .venv
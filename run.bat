@echo off
REM VaultGuard — Windows convenience launcher
REM Usage: run.bat [setup|train|importance|app]

IF "%1"=="setup"      pip install -e .
IF "%1"=="train"      python -m src.train
IF "%1"=="importance" python -m src.feature_importance
IF "%1"=="app"        streamlit run app.py
IF "%1"==""           streamlit run app.py

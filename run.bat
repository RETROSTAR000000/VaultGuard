@echo off
REM ============================================================
REM  VaultGuard — Windows convenience launcher (uses uv)
REM  Usage: run.bat [setup|train|importance|app|notebook|clean]
REM ============================================================
setlocal

REM ── locate uv ──────────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo [INFO] uv not found. Installing via PowerShell...
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo [ERROR] uv installation failed. Please install manually:
        echo         https://docs.astral.sh/uv/getting-started/installation/
        exit /b 1
    )
    REM Refresh PATH so uv is visible in this session
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

REM ── route commands ─────────────────────────────────────────
IF /I "%1"=="setup"      goto :setup
IF /I "%1"=="train"      goto :train
IF /I "%1"=="importance" goto :importance
IF /I "%1"=="app"        goto :app
IF /I "%1"=="notebook"   goto :notebook
IF /I "%1"=="clean"      goto :clean
IF "%1"==""              goto :app

echo [ERROR] Unknown command: %1
echo Usage: run.bat [setup^|train^|importance^|app^|notebook^|clean]
exit /b 1

:setup
echo [INFO] Creating virtual environment and installing dependencies...
uv sync
goto :done

:train
echo [INFO] Training fraud models...
uv run python -m src.train
goto :done

:importance
echo [INFO] Computing feature importance...
uv run python -m src.feature_importance
goto :done

:app
echo [INFO] Starting VaultGuard Streamlit app...
uv run streamlit run app.py
goto :done

:notebook
echo [INFO] Starting Jupyter...
uv run jupyter notebook notebooks/
goto :done

:clean
echo [INFO] Removing virtual environment...
if exist .venv rmdir /s /q .venv
echo [INFO] Done.
goto :done

:done
endlocal

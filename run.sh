#!/usr/bin/env bash
# ============================================================
#  VaultGuard — macOS / Linux convenience launcher (uses uv)
#  Usage: ./run.sh [setup|train|importance|app|notebook|clean]
# ============================================================
set -euo pipefail

# ── locate / install uv ─────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found. Installing via curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session (installer puts it in ~/.local/bin)
    export PATH="$HOME/.local/bin:$PATH"
fi

COMMAND="${1:-app}"

case "$COMMAND" in
    setup)
        echo "[INFO] Creating virtual environment and installing dependencies..."
        uv sync
        ;;
    train)
        echo "[INFO] Training fraud models..."
        uv run python -m src.train
        ;;
    importance)
        echo "[INFO] Computing feature importance..."
        uv run python -m src.feature_importance
        ;;
    app)
        echo "[INFO] Starting VaultGuard Streamlit app..."
        uv run streamlit run app.py
        ;;
    notebook)
        echo "[INFO] Starting Jupyter..."
        uv run jupyter notebook notebooks/
        ;;
    clean)
        echo "[INFO] Removing virtual environment..."
        rm -rf .venv
        echo "[INFO] Done."
        ;;
    *)
        echo "[ERROR] Unknown command: $COMMAND"
        echo "Usage: $0 [setup|train|importance|app|notebook|clean]"
        exit 1
        ;;
esac

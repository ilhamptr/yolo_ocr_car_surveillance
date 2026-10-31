#!/usr/bin/env bash
# Run the parking tracker locally (no Docker). Starts the API and the UI,
# then stops both cleanly on Ctrl+C.
#
# Usage: ./run_local.sh
set -e

cd "$(dirname "$0")"

PY=./env/bin
export VIDEOS_DIR=./videos
export DB_PATH=./parking.db
export API_URL=http://127.0.0.1:8000

echo "Starting API on :8000 ..."
$PY/uvicorn app.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Stop the API when this script exits (Ctrl+C or UI quit).
cleanup() {
  echo ""
  echo "Stopping API (pid $API_PID) ..."
  kill $API_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting UI on :8501 ..."
echo "Open http://127.0.0.1:8501 in your browser."
$PY/streamlit run app/ui.py --server.address 127.0.0.1 --server.port 8501

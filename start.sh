#!/bin/bash

# Adaptive AI Trading Bot - Unified Startup Script

echo "====================================================="
echo "  Starting Adaptive AI Trading Bot System"
echo "====================================================="

# Navigate to the script's directory (to allow running from anywhere)
cd "$(dirname "$0")"

# Clean up old listeners so the UI does not attach to stale dev servers
for PORT in 8000 5173; do
    EXISTING_PIDS=$(lsof -ti tcp:$PORT)
    if [ -n "$EXISTING_PIDS" ]; then
        echo "[System] Clearing existing process(es) on port $PORT: $EXISTING_PIDS"
        kill $EXISTING_PIDS 2>/dev/null
        sleep 1
    fi
done

# 1. Start the Python Backend
echo "[System] Booting PyTorch Backend & WebSocket Engine..."
cd backend
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "[Error] Python virtual environment not found in backend/venv."
    echo "Please ensure you have run 'python3 -m venv venv' and installed requirements."
    exit 1
fi

uvicorn main:app --host 127.0.0.1 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "[System] Backend running on port 8000 (PID: $BACKEND_PID)"
cd ..

# 2. Start the React Frontend
echo "[System] Booting React Dashboard..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "[System] Installing npm dependencies (first time run)..."
    npm install > /dev/null 2>&1
fi

npm run dev -- --host 127.0.0.1 --port 5173 > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "[System] Frontend running on port 5173 (PID: $FRONTEND_PID)"
cd ..

# Wait briefly to ensure servers are up
sleep 3

# 3. Open the Dashboard in the default browser
echo "[System] Launching Dashboard in web browser..."
if command -v open > /dev/null; then
    open "http://localhost:5173" # macOS
elif command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:5173" # Linux
elif command -v start > /dev/null; then
    start "http://localhost:5173" # Windows
else
    echo "Please open http://localhost:5173 in your browser."
fi

echo "====================================================="
echo "  System is LIVE. Press Ctrl+C to shut down."
echo "====================================================="

# Trap Ctrl+C (SIGINT) to gracefully kill both background servers
trap "echo -e '\n[System] Shutting down servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

# Keep script active to catch Ctrl+C
wait $BACKEND_PID $FRONTEND_PID

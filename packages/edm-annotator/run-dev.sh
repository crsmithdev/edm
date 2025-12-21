#!/bin/bash
# Development server launcher for EDM Annotator

set -e

# ANSI color codes
BOLD="\033[1m"
GREEN="\033[92m"
BLUE="\033[94m"
CYAN="\033[96m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"
DIM="\033[2m"

# Logging function with timestamp
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Stream processor: adds timestamp and source tag to each line
stream_with_timestamp() {
    local source="$1"
    while IFS= read -r line; do
        # Remove leading/trailing whitespace
        line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        # Strip existing timestamp patterns (e.g., "8:04:59 AM", "08:04:59")
        line=$(echo "$line" | sed 's/^[0-9]\{1,2\}:[0-9]\{2\}:[0-9]\{2\}\( [AP]M\)\{0,1\}[[:space:]]*//')
        # Only print non-empty lines
        if [ -n "$line" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') [$source] $line"
        fi
    done
}

log "${BOLD}${CYAN}EDM Structure Annotator - Development Mode${RESET}"
echo ""

# Kill any existing processes on port 5001
EXISTING_PIDS=$(lsof -ti :5001 2>/dev/null || true)
if [ -n "$EXISTING_PIDS" ]; then
    log "${YELLOW}Cleaning up existing processes on port 5001${RESET}"
    echo "$EXISTING_PIDS" | xargs kill -9 2>/dev/null
    sleep 1
fi

# Check if backend is set up
if ! uv run edm-annotator --help &> /dev/null; then
    log "${RED}Error: Backend not installed. Run: uv sync${RESET}"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    log "${RED}Error: Frontend dependencies not installed. Run: cd frontend && npm install${RESET}"
    exit 1
fi

log "${BOLD}Starting Servers${RESET}"
log "Backend (Flask)  → ${GREEN}http://localhost:5001${RESET}"
log "Frontend (Vite)  → ${GREEN}http://localhost:5174${RESET}"
log "Starting backend..."

# Function to cleanup background processes
cleanup() {
    # Prevent recursive trap execution
    trap - SIGINT SIGTERM EXIT

    log "${BOLD}Shutting Down${RESET}"
    log "Stopping backend (PID: $BACKEND_PID)"
    kill $BACKEND_PID 2>/dev/null
    log "Stopping frontend (PID: $FRONTEND_PID)"
    kill $FRONTEND_PID 2>/dev/null
    log "${GREEN}Servers stopped${RESET}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start backend in background
uv run edm-annotator --env development --port 5001 2>&1 | stream_with_timestamp "backend" &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2
log "${GREEN}Backend running (PID: ${BACKEND_PID})${RESET}"
log "Starting frontend..."

# Start frontend in background
cd frontend
npm run dev 2>&1 | stream_with_timestamp "frontend" &
FRONTEND_PID=$!
cd ..

sleep 1
log "${GREEN}Frontend running (PID: ${FRONTEND_PID})${RESET}"
log "${GREEN}Ready! Open ${CYAN}http://localhost:5174${RESET} in your browser${RESET}"
log "Press Ctrl+C to stop both servers"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

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

echo -e "\n${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║${RESET}  ${BOLD}🎵  EDM Structure Annotator - Development Mode${RESET}      ${BOLD}${CYAN}║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}\n"

echo -e "${BOLD}${BLUE}Checking Prerequisites${RESET}"

# Check if backend is set up
if ! uv run edm-annotator --help &> /dev/null; then
    echo -e "  ${RED}✗${RESET} Backend not installed"
    echo -e "    ${DIM}Run: ${CYAN}uv sync${RESET}"
    exit 1
fi
echo -e "  ${GREEN}✓${RESET} Backend installed"

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo -e "  ${RED}✗${RESET} Frontend dependencies not installed"
    echo -e "    ${DIM}Run: ${CYAN}cd frontend && npm install${RESET}"
    exit 1
fi
echo -e "  ${GREEN}✓${RESET} Frontend dependencies installed\n"

echo -e "${BOLD}${BLUE}Starting Servers${RESET}"
echo -e "  ${DIM}┌────────────────────────────────────────────────────────┐${RESET}"
echo -e "  ${DIM}│${RESET} Backend (Flask)    ${GREEN}http://localhost:5000${RESET}               ${DIM}│${RESET}"
echo -e "  ${DIM}│${RESET} Frontend (Vite)    ${GREEN}http://localhost:5174${RESET}               ${DIM}│${RESET}"
echo -e "  ${DIM}└────────────────────────────────────────────────────────┘${RESET}\n"

echo -e "  ${DIM}Starting backend...${RESET}"

# Function to cleanup background processes
cleanup() {
    echo -e "\n${BOLD}${BLUE}Shutting Down${RESET}"
    echo -e "  ${DIM}Stopping backend (PID: $BACKEND_PID)${RESET}"
    kill $BACKEND_PID 2>/dev/null
    echo -e "  ${DIM}Stopping frontend (PID: $FRONTEND_PID)${RESET}"
    kill $FRONTEND_PID 2>/dev/null
    echo -e "  ${GREEN}✓${RESET} Servers stopped\n"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
uv run edm-annotator --env development --port 5000 2>&1 | sed "s/^/  ${DIM}[backend]${RESET}  /" &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2
echo -e "  ${GREEN}✓${RESET} Backend running (PID: ${BACKEND_PID})\n"

echo -e "  ${DIM}Starting frontend...${RESET}"

# Start frontend in background
cd frontend
npm run dev 2>&1 | sed "s/^/  ${DIM}[frontend]${RESET} /" &
FRONTEND_PID=$!
cd ..

sleep 1
echo -e "  ${GREEN}✓${RESET} Frontend running (PID: ${FRONTEND_PID})\n"

echo -e "${BOLD}${GREEN}Ready!${RESET} Open ${BOLD}${CYAN}http://localhost:5174${RESET} in your browser\n"
echo -e "${DIM}Press Ctrl+C to stop both servers${RESET}\n"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

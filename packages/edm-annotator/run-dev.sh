#!/bin/bash
# Development server launcher for EDM Annotator

set -e

echo "🎵 EDM Structure Annotator - Development Mode"
echo "=============================================="
echo ""

# Check if backend is set up
if ! uv run edm-annotator --help &> /dev/null; then
    echo "❌ Backend not installed. Run:"
    echo "   uv sync"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not installed. Run:"
    echo "   cd frontend && npm install"
    exit 1
fi

echo "✅ Prerequisites met"
echo ""
echo "Starting servers..."
echo "  Backend:  http://localhost:5000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend in background
uv run edm-annotator --env development --port 5000 &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start frontend in background
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

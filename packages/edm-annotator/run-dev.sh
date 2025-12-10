#!/bin/bash
# Development server launcher for EDM Annotator

set -e

echo "🎵 EDM Structure Annotator - Development Mode"
echo "=============================================="
echo ""

# Check if backend is set up
if ! command -v edm-annotator &> /dev/null; then
    echo "❌ Backend not installed. Run:"
    echo "   cd backend && uv pip install -e \".[dev]\""
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not installed. Run:"
    echo "   cd frontend && pnpm install"
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
cd backend
edm-annotator --env development --port 5000 &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 2

# Start frontend in background
cd frontend
pnpm dev &
FRONTEND_PID=$!
cd ..

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

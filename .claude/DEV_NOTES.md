# EDM Development Notes

Quick reference for common development tasks.

## Running the Annotator Web App

The EDM annotator is a React + Flask application for annotating track structures.

### Quick Start

```bash
# From repo root
just annotator
```

This starts:
- Backend API: http://localhost:5000 (Flask)
- Frontend: http://localhost:5173 (Vite + React + TypeScript)

### First Time Setup

```bash
# Install backend (from repo root)
uv sync

# Install frontend
cd packages/edm-annotator/frontend
npm install
```

### Manual Start (Alternative)

If you need to run servers separately:

```bash
# Terminal 1: Backend
cd packages/edm-annotator
uv run edm-annotator --env development --port 5000

# Terminal 2: Frontend
cd packages/edm-annotator/frontend
npm run dev
```

## Package Structure

- **packages/edm-annotator/** - Annotator package root
  - **pyproject.toml** - Python package config (v2.0.0)
  - **backend/src/edm_annotator/** - Flask API
  - **frontend/src/** - React app (TypeScript + Vite)
  - **run-dev.sh** - Dev server launcher script

## Common Issues

### Backend won't start
- Run `uv sync` from repo root
- Verify with: `uv run edm-annotator --help`

### Frontend won't start
- Run `npm install` in packages/edm-annotator/frontend
- Check Node version: `node --version` (needs 18+)

### Port already in use
- Backend: Change port with `--port 5001`
- Frontend: Vite will auto-increment (5174, 5175, etc.)

## Documentation

- **packages/edm-annotator/QUICKSTART.md** - Detailed setup guide
- **packages/edm-annotator/README.md** - Full documentation
- **justfile** - All available commands (run `just` to list)

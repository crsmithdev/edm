# Troubleshooting Guide

## Tracks Not Loading (500 Internal Server Error)

### Symptoms
- Frontend shows "Failed to load resource: the server responded with a status of 500 (INTERNAL SERVER ERROR)"
- Tracks list is empty or fails to load
- Backend logs show `ModuleNotFoundError: No module named 'edm.data'`

### Root Cause
The backend depends on the `edm` package (edm-lib) which must be installed via the workspace. If you run `uv sync` from inside the `packages/edm-annotator/backend` directory, it won't properly link the workspace packages.

### Solution
Always run `uv sync` from the **workspace root** (`/home/crsmi/edm`), not from inside package directories:

```bash
# ✅ CORRECT: From workspace root
cd /home/crsmi/edm
uv sync

# ❌ WRONG: From inside backend directory
cd /home/crsmi/edm/packages/edm-annotator/backend
uv sync  # This won't link workspace packages properly!
```

### Verification
Test that the app can import successfully:

```bash
# From workspace root
uv run python -c "from edm_annotator.app import create_app; print('✓ OK')"
```

If you see `ModuleNotFoundError: No module named 'edm.data'`, you need to run `uv sync` from the workspace root.

### Prevention
The test suite includes regression tests for this issue:

```bash
# Run import validation tests
uv run pytest packages/edm-annotator/backend/tests/test_app_imports.py -v
```

These tests will fail if the `edm-lib` dependency is not properly installed, catching the issue before runtime.

## Other Common Issues

### Port Already in Use
The dev server (`run-dev.sh`) now automatically cleans up any processes using port 5000 before starting. If you still encounter issues:

```bash
# Manually check what's using port 5000
lsof -i :5000

# The dev server will automatically kill orphaned processes
# But you can also do it manually if needed:
lsof -ti :5000 | xargs kill -9
```

**Note**: Previous versions of the dev server could leave orphaned processes when terminated forcefully. This has been fixed by:
- Automatic port cleanup on startup
- Improved trap handling (SIGINT, SIGTERM, EXIT)

### Frontend Dependencies Missing
If the dev server script reports "Frontend dependencies not installed":

```bash
cd packages/edm-annotator/frontend
npm install
```

### Backend Not Found
If you see "Backend not installed" errors:

```bash
# From workspace root
uv sync
```

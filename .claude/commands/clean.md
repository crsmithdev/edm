Clean build artifacts and caches.

```bash
echo "Cleaning Python caches..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

echo "Cleaning coverage files..."
rm -rf .coverage htmlcov/ 2>/dev/null || true

echo "Cleaning build artifacts..."
rm -rf dist/ build/ *.egg-info 2>/dev/null || true

echo "✓ Cleaned all build artifacts and caches"
```

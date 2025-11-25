Pull latest changes and sync dependencies.

```bash
set -e
echo "Pulling latest changes..."
git pull
echo "Syncing dependencies..."
uv sync
echo "✓ Repository and dependencies synchronized"
```

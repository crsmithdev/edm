Display comprehensive project status.

```bash
echo "=== Git Status ==="
git status --short

echo ""
echo "=== OpenSpec Changes ==="
openspec list 2>/dev/null || echo "No OpenSpec changes"

echo ""
echo "=== TODOs ==="
if [ -f TODO.md ]; then
  todo_count=$(grep -c "^- \[ \]" TODO.md 2>/dev/null || echo "0")
  echo "Open tasks: $todo_count"
else
  echo "No TODO.md file"
fi

echo ""
echo "=== Ideas ==="
if [ -f IDEAS.md ]; then
  # Count ideas (lines starting with "- **")
  idea_count=$(grep -c "^- \*\*" IDEAS.md 2>/dev/null || echo "0")
  echo "Captured ideas: $idea_count"
else
  echo "No IDEAS.md file"
fi
```

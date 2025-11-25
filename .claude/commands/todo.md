Add a task to TODO.md interactively.

```bash
#!/bin/bash

# Create TODO.md if it doesn't exist
if [ ! -f TODO.md ]; then
  cat > TODO.md <<'TEMPLATE'
# TODO

## High Priority

## Medium Priority

## Low Priority / Nice to Have
TEMPLATE
fi

# Prompt for priority
echo "Priority?"
echo "1) High"
echo "2) Medium"
echo "3) Low"
read -p "Select (1-3): " priority_choice

case $priority_choice in
  1) priority="High Priority" ;;
  2) priority="Medium Priority" ;;
  3) priority="Low Priority / Nice to Have" ;;
  *) priority="Medium Priority" ;;
esac

# Prompt for task
read -p "Task description: " task

# Add to TODO.md under appropriate section
awk -v pri="## $priority" -v task="- [ ] $task" '
  /^## / { in_section = ($0 == pri) }
  { print }
  in_section && /^## / && !added { print task; added=1; in_section=0 }
' TODO.md > TODO.md.tmp && mv TODO.md.tmp TODO.md

echo "✓ Added to TODO.md under $priority"
```

Add an improvement idea to IDEAS.md interactively.

```bash
#!/bin/bash

# Create IDEAS.md if it doesn't exist
if [ ! -f IDEAS.md ]; then
  cat > IDEAS.md <<'TEMPLATE'
# Improvement Ideas

## Ready for Proposal
Ideas that should become OpenSpec proposals soon.

## Under Consideration
Ideas that need more investigation.

## Icebox
Good ideas but low priority.
TEMPLATE
fi

# Prompt for category
echo "Category?"
echo "1) Ready for Proposal"
echo "2) Under Consideration"
echo "3) Icebox"
read -p "Select (1-3): " category_choice

case $category_choice in
  1) category="Ready for Proposal" ;;
  2) category="Under Consideration" ;;
  3) category="Icebox" ;;
  *) category="Under Consideration" ;;
esac

# Prompt for description
read -p "Idea description: " idea
read -p "Context/source (optional): " context

# Build idea entry
timestamp=$(date +%Y-%m-%d)
idea_entry="- **$idea**"
if [ -n "$context" ]; then
  idea_entry="$idea_entry\n  - Source: $context\n  - Added: $timestamp"
else
  idea_entry="$idea_entry\n  - Added: $timestamp"
fi

# Add to IDEAS.md under appropriate section
awk -v cat="## $category" -v idea="$idea_entry" '
  /^## / { in_section = ($0 == cat) }
  { print }
  in_section && /^## / && !added { print idea; print ""; added=1; in_section=0 }
' IDEAS.md > IDEAS.md.tmp && mv IDEAS.md.tmp IDEAS.md

echo "✓ Added to IDEAS.md under $category"
```

#!/usr/bin/env python3
"""Migrate existing annotations to tier-based subdirectories."""

import shutil
from pathlib import Path

import yaml

ANNOTATION_DIR = Path("data/annotations")
REFERENCE_DIR = ANNOTATION_DIR / "reference"
GENERATED_DIR = ANNOTATION_DIR / "generated"

# Create subdirectories
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# Process all YAML files in root directory
moved = {"reference": 0, "generated": 0, "skipped": 0}

for yaml_path in sorted(ANNOTATION_DIR.glob("*.yaml")):
    try:
        # Read tier from metadata
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        tier = data.get("metadata", {}).get("tier", 2)

        # Determine destination
        if tier == 1:
            dest = REFERENCE_DIR / yaml_path.name
            moved["reference"] += 1
        else:  # tier 2 or 3
            dest = GENERATED_DIR / yaml_path.name
            moved["generated"] += 1

        # Move file
        shutil.move(str(yaml_path), str(dest))
        print(f"Moved {yaml_path.name} → {dest.parent.name}/")

    except Exception as e:
        print(f"⚠️  Failed to process {yaml_path.name}: {e}")
        moved["skipped"] += 1

print("\n✓ Migration complete:")
print(f"  Reference: {moved['reference']} files")
print(f"  Generated: {moved['generated']} files")
print(f"  Skipped: {moved['skipped']} files")

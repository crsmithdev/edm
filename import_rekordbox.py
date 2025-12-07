#!/usr/bin/env python3
"""Import Rekordbox XML file into annotations."""

from pathlib import Path

from edm.data.converters import batch_convert_rekordbox_xml
from edm.data.metadata import AnnotationTier

# Paths
xml_path = Path("/mnt/c/Music/music.xml")
output_dir = Path("data/annotations/generated")

print(f"Importing Rekordbox XML from: {xml_path}")
print(f"Output directory: {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)
print("Configuration: Hot cues only, deduplicate=True")
print()

# Convert with hot cues only (avoids duplicates with memory cues)
success, skipped, errors = batch_convert_rekordbox_xml(
    xml_path=xml_path,
    output_dir=output_dir,
    tier=AnnotationTier.AUTO_CLEANED,
    confidence=0.8,
    skip_existing=False,
)

# Report results
print(f"✓ Successfully imported: {success} tracks")
print(f"⊘ Skipped (already exists): {skipped} tracks")

if errors:
    print(f"\n✗ Errors ({len(errors)}):")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ No errors")

print(f"\nTotal annotations in output dir: {len(list(output_dir.glob('*.yaml')))}")

#!/usr/bin/env python3
"""Import Rekordbox XML file into JAMS format (MIR standard)."""

from pathlib import Path

from edm.data.jams_io import batch_convert_rekordbox_to_jams
from edm.data.metadata import AnnotationTier

# Paths
xml_path = Path("/mnt/c/Music/music.xml")
output_dir = Path("data/jams")

print("Importing Rekordbox XML to JAMS format")
print(f"Source: {xml_path}")
print(f"Output: {output_dir}")
print("Configuration: Hot cues only, deduplicate=True")
print()

# Convert with hot cues only (avoids duplicates with memory cues)
success, skipped, errors = batch_convert_rekordbox_to_jams(
    xml_path=xml_path,
    output_dir=output_dir,
    tier=AnnotationTier.AUTO_CLEANED,
    confidence=0.8,
    skip_existing=True,
    cue_types=["hot"],  # Only import hot cues (Num >= 0)
    deduplicate=True,  # Remove any remaining duplicates
)

# Report results
print(f"✓ Successfully imported: {success} tracks")
print(f"⊘ Skipped (already exists): {skipped} tracks")

if errors:
    print(f"\n✗ Errors ({len(errors)}):")
    for error in errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")
else:
    print("✓ No errors")

print(f"\nTotal JAMS files in output dir: {len(list(output_dir.glob('*.jams')))}")

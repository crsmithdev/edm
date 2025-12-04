# Design: Workflow Restructure

## Approach

Three-layer change: slash commands → shell scripts → CLI integration.

### 1. Slash Commands

**`/analyze`** (new)
- Command: `.claude/scripts/analyze.sh [files...]`
- Behavior: Run `uv run edm analyze --annotations FILE -o data/generated/SANITIZED.yaml` per file
- No args: error (explicit file paths required)

**`/annotate`** (modified)
- Command: `.claude/scripts/annotate.sh`
- Behavior: Merge `data/generated/*.yaml` into `data/annotations/*.yaml`
- Logic: For each generated file, if annotation exists → append raw as comments; else copy full file

**`/evaluate`** (new)
- Command: `.claude/scripts/evaluate.sh [files...]`
- Behavior: Run `uv run edm evaluate` with appropriate args
- No args: evaluate all tracks in both directories

### 2. Shell Scripts

**`.claude/scripts/analyze.sh`**
```bash
#!/bin/bash
# Analyze audio files → data/generated/

for file in "$@"; do
    name=$(sanitize_name "$(basename "$file")")
    uv run edm analyze --annotations "$file" -o "data/generated/${name}.yaml"
done
```

**`.claude/scripts/annotate.sh`**
```bash
#!/bin/bash
# Merge generated → annotations (preserve user data)

for gen_file in data/generated/*.yaml; do
    name=$(basename "$gen_file")
    anno_file="data/annotations/${name}"

    if [ -f "$anno_file" ]; then
        # Extract raw from generated, append to annotation
        strip_raw "$anno_file" > temp.yaml
        extract_raw "$gen_file" >> temp.yaml
        mv temp.yaml "$anno_file"
    else
        # New file: copy with raw as comments
        cp "$gen_file" "$anno_file"
    fi
done
```

**`.claude/scripts/evaluate.sh`**
```bash
#!/bin/bash
# Evaluate analysis accuracy

if [ $# -gt 0 ]; then
    # Specific files
    uv run edm evaluate --files "$@"
else
    # All annotated tracks
    uv run edm evaluate
fi
```

### 3. Context Consolidation

**`.claude/contexts/evaluation.xml`** (replaces `annotate.xml`, `evaluate.xml`)

```xml
<evaluation-context>
  <commands>
    <cmd name="/analyze" args="[files]">Analyze audio → data/generated/</cmd>
    <cmd name="/annotate">Merge generated → annotations</cmd>
    <cmd name="/evaluate" args="[files?]">Compare analysis accuracy</cmd>
  </commands>

  <directories>
    <dir path="data/generated/" tracked="no" role="analysis output"/>
    <dir path="data/annotations/" tracked="git" role="ground truth"/>
    <dir path="~/music" role="test audio files"/>
  </directories>

  <cli>
    <analyze cmd="uv run edm analyze --annotations FILE -o OUTPUT.yaml"/>
    <evaluate cmd="uv run edm evaluate [--files FILES]"/>
    <flags annotations="structured YAML output" json="machine-readable"/>
  </cli>

  <output-format>
    <structure>file, duration, bpm, downbeat, time_signature, structure, raw</structure>
    <raw>Per-detector sections: msaf, energy, future detectors</raw>
    <annotation>Clean structure + commented raw after --- separator</annotation>
  </output-format>

  <evaluation>
    <metrics>boundary-f1 (±2s tolerance), label-accuracy, precision, recall</metrics>
    <bar-to-time>downbeat + (bar-1) * 60/bpm * 4</bar-to-time>
  </evaluation>

  <analysis-workflow>
    <bpm cascade="metadata → beat_this → librosa"/>
    <structure cascade="msaf → energy fallback"/>
    <detectors msaf="spectral flux + EDM labels" energy="RMS + contrast"/>
  </analysis-workflow>

  <safety>
    <rule>Never delete from data/annotations/</rule>
    <rule>Merge preserves user edits to structure field</rule>
    <rule>Raw data in comments, never user-edited</rule>
  </safety>
</evaluation-context>
```

## Implementation

### File Changes

**New:**
- `.claude/scripts/analyze.sh`
- `.claude/scripts/evaluate.sh`
- `.claude/contexts/evaluation.xml`
- `.claude/commands/analyze.md`
- `.claude/commands/evaluate.md`

**Modified:**
- `.claude/scripts/annotate.sh` (simplified: remove generate/merge/save subcommands)
- `.claude/commands/annotate.md` (update to reference evaluation context)

**Deleted:**
- `.claude/contexts/annotate.xml`
- `.claude/contexts/evaluate.xml`

### edm CLI Verification

Check current `edm analyze` output format:
1. Run `uv run edm analyze --help` to verify `--annotations` flag
2. Inspect YAML output structure for raw data inclusion
3. If raw data missing: update `src/cli/commands/analyze.py` to include per-detector sections

## Testing

### Manual Testing

1. **Analysis**: `/analyze ~/music/test.flac` → verify `data/generated/test.yaml` contains raw sections
2. **Annotation (new file)**: `/annotate` → verify new file created in `data/annotations/`
3. **Annotation (merge)**: Edit structure in annotation, re-analyze, `/annotate` → verify structure preserved, raw updated
4. **Evaluation**: `/evaluate` → verify F-scores computed against all annotations

### Edge Cases

- **Empty generated/**: `/annotate` should no-op gracefully
- **Missing audio file**: `/analyze missing.flac` should error clearly
- **Corrupted YAML**: Shell scripts should validate YAML before overwriting annotations

## Risks

**Data loss:**
- Script bugs could overwrite user annotations (test extensively)

**CLI incompatibility:**
- `edm analyze` may not support `--annotations` or raw output format
- Mitigation: Verify CLI capabilities; update analyze.py if needed

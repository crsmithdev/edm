---
status: draft
created: 2025-12-03
---

# CLI Simplification - Design

## Approach

Surgical removal of unused features and consolidation of output/logging mechanisms. Follow Unix philosophy: do one thing well, compose via pipes.

## Implementation

### File Changes

**src/cli/commands/analyze.py**
- Remove argument definitions: `--offline`, `--json-logs`, `--annotations`, `--structure-detector`
- Rename: `--ignore-metadata` → `--no-metadata`
- Replace: `--log-level` with `-v`/`-vv` handling (use count)
- Replace: `--no-color` with proper terminal detection + explicit disable flag
- Replace: `--output` with stdout-only output (remove file writing logic)
- Fix: `--workers` logic to cap at min(max_workers, file_count)

**src/cli/commands/evaluate.py**
- Adjust: `--reference` to handle glob patterns (use `pathlib.Path.glob()`)
- Remove: `--tolerance`, `--detector` arguments
- Replace: `--output` with stdout-only
- Set defaults in code: tolerance=2.0s, detector="msaf"

**src/cli/commands/profile.py**
- Delete file entirely

**src/edm/profiling/**
- Delete directory

**src/cli/main.py**
- Remove `/profile` command registration

### Logging Configuration

```python
# Map verbosity count to log level
verbosity_to_level = {
    0: logging.WARNING,  # default
    1: logging.INFO,     # -v
    2: logging.DEBUG,    # -vv
}
level = verbosity_to_level.get(args.verbose, logging.DEBUG)
logging.basicConfig(level=level)
```

### Output Handling

Remove all `--output` file writing logic. Users pipe:
```bash
edm analyze track.flac > results.txt
edm evaluate > eval_report.txt
```

Ensure CLI output goes to stdout, logs go to stderr.

### Workers Fix

```python
actual_workers = min(args.workers, len(files)) if args.workers else min(os.cpu_count(), len(files))
```

## Testing

- Verify removed flags error appropriately
- Test `-v` and `-vv` produce expected log levels
- Confirm `--no-metadata` works like old `--ignore-metadata`
- Test output redirection: `edm analyze track.flac > out.txt`
- Verify workers don't over-allocate for small file counts

## Risks

- Breaking scripts that use removed flags (low usage expected)
- Output format changes if stdout/stderr separation was incorrect before

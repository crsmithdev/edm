---
status: deployed
created: 2025-12-03
updated: 2025-12-03
archived: 2025-12-03
---

# [CLI]Simplify Interface

## Why

The CLI has accumulated technical debt and unused features:
- Unused flags from earlier iterations (`--offline`, `--json-logs`, `--annotations`)
- Overly granular configuration that should have sensible defaults (`--structure-detector`, `--tolerance`)
- Inconsistent output mechanisms (`--output` for file writing vs proper stdout piping)
- Unnecessarily verbose logging control (`--log-level` vs standard `-v`/`-vv`)
- Profiling infrastructure no longer needed (`/profile` command, `src/edm/profiling/`)
- Worker allocation that spins up max workers regardless of file count

This creates friction for users and maintenance burden for developers.

## What

**Affected Specs**
- `openspec/specs/cli/spec.md` - Command-line interface
- `openspec/specs/analysis/spec.md` - Analysis functionality

**Changes Required**

### `analyze` subcommand
- **Remove**: `--json-logs` (unused) ✓
- **Remove**: `--structure-detector` (always use msaf) ✓
- **Rename**: `--ignore-metadata` → `--no-metadata` ✓
- **Replace**: `--log-level` → standard `-v`/`-vv` flags ✓
  - Default: warning-level logging
  - `-v`/`--verbose`: info-level logging
  - `-vv`: debug-level logging
- **Fix**: `--workers` → adjust to not spin up max workers when file count is lower ✓
- **Keep**: `--offline` (used by workflow)
- **Keep**: `--annotations` (used by workflow)
- **Keep**: `--output` (used by workflow)
- **Keep**: `--no-color` (working as intended)

### `evaluate` subcommand
- **Remove**: `--tolerance` (hardcode default 2.0s) ✓
- **Remove**: `--detector` (hardcode msaf) ✓
- **Keep**: `--output` (used by workflow)
- **Keep**: `--reference` (working as intended)

### `/profile` command
- **Remove**: Entire command
- **Remove**: `src/edm/profiling/` directory

## Impact

**Breaking Changes**
- Users relying on removed flags will need to update scripts
- Migration: none of the removed flags appear to be widely used

**Benefits**
- Cleaner, more intuitive CLI surface
- Reduced code maintenance burden
- Better alignment with Unix philosophy (pipe output to files)
- More consistent with standard CLI conventions (e.g., `-v` verbosity)

**Risks**
- Low risk: most removed features appear unused
- Ensure appropriate defaults are chosen for removed configurability

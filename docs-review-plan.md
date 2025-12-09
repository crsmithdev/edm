# Documentation Review Plan

## Scope
Documentation review focused on accuracy, completeness, and effectiveness for both users and AI agents.

## Current State Assessment

### Documentation Files Reviewed
- README.md (104 lines)
- docs/cli-reference.md (268 lines)
- docs/architecture.md (364 lines)
- docs/development.md (288 lines)
- docs/agent-guide.md (113 lines)
- docs/testing.md (247 lines)
- docs/project-structure.md (111 lines)

### Additional Documentation
- docs/python-style.md
- docs/edm-terminology.md
- docs/learning-based-pivot-research.md
- docs/analysis-algorithms.md
- docs/training.md
- docs/experiment-tracking.md
- docs/mlops.md
- docs/week1-summary.md
- docs/week1-final-summary.md

## Key Findings

### 1. README.md - ACCURATE with minor gaps

**Strengths:**
- Clear installation instructions
- Accurate BPM detection strategy description
- Good quick start examples
- Performance benchmarks present
- Links to detailed docs

**Issues:**
- Missing `analyze` command in CLI help output (shows: evaluate, train, analyze, data, models)
- README shows `--offline` flag but CLI help doesn't list it
- README structure analysis examples missing (only BPM shown)
- No mention of `data` or `models` commands that exist in CLI
- Missing `train` command documentation

**Recommendations:**
1. Add structure analysis examples to Quick Start
2. Document all CLI commands (train, data, models) or note they're advanced/experimental
3. Verify --offline flag exists or remove from examples
4. Add section on available analysis types (bpm, beats, grid, structure)

### 2. CLI Reference (docs/cli-reference.md) - MOSTLY ACCURATE with discrepancies

**Issues:**
- Documents `--structure-detector` option (auto/msaf/energy) - not visible in --help
- Shows `--log-level` as option - not in current --help (uses `-v`/`-vv` instead)
- Shows `--json-logs` flag - not in current --help
- Documents `--offline` flag - not in current --help
- Missing `--cache-size` option (present in --help)
- Missing `-vv` verbosity levels
- Missing `data` and `models` commands entirely
- Missing `train` command

**Recommendations:**
1. Run actual CLI and verify ALL options match reality
2. Add data/models/train commands or note as experimental
3. Update verbosity documentation (-v/-vv instead of --log-level)
4. Verify structure detector options still exist
5. Document cache-size option

### 3. Architecture Documentation (docs/architecture.md) - EXCELLENT

**Strengths:**
- Clear two-tier abstraction pattern explained
- Good module organization diagram
- Design decisions well-documented
- Data flow diagrams helpful
- Placeholder features clearly marked

**Issues:**
- References "madmom" in some places but beat_this is actual default
- "use_madmom" parameter name in config is legacy (should be use_beat_this)
- MSAF integration details may be outdated (need to verify current implementation)

**Recommendations:**
1. Search codebase for madmom references and update to beat_this
2. Verify MSAF is still used (architecture says it is)
3. Update legacy parameter names in documentation

### 4. Development Guide (docs/development.md) - GOOD

**Strengths:**
- Clear setup instructions
- Good test examples
- Logging patterns documented
- Evaluation framework coverage

**Minor gaps:**
- No mention of parallel processing options (--workers)
- Missing data/models/train commands

### 5. Agent Guide (docs/agent-guide.md) - EFFECTIVE

**Strengths:**
- Task-based navigation
- Good file locations
- Common commands section

**Potential improvements:**
- Could include more about ML training workflow
- Could document data management workflow

### 6. Testing Documentation (docs/testing.md) - COMPREHENSIVE

**Strengths:**
- Complete test structure
- Good examples
- Coverage information
- Fixture documentation

**No major issues identified**

### 7. CLAUDE.md Status

**Finding:** No CLAUDE.md in project root
- This is mentioned in global ~/.claude/CLAUDE.md but doesn't exist at project level
- May be intentional (using only global CLAUDE.md)

**Recommendation:** Clarify if project CLAUDE.md is needed or confirm global-only approach is sufficient

## Critical Issues (Fix Immediately)

### C1: CLI Reference Out of Sync
**Impact:** Users follow incorrect documentation, commands fail
**Files:** docs/cli-reference.md, README.md
**Action:**
1. Run `edm --help`, `edm analyze --help`, `edm evaluate --help` for all commands
2. Update docs to match exact current CLI
3. Remove references to non-existent flags (--offline, --log-level, --json-logs, --structure-detector)
4. Add missing flags (--cache-size, -v/-vv verbosity)

### C2: Legacy "madmom" terminology
**Impact:** Confusion about what library is actually used
**Files:** docs/architecture.md, docs/cli-reference.md, likely config.py
**Action:**
1. Search all docs for "madmom" references
2. Update to "beat_this" or note as legacy name
3. Update config parameter names or document legacy names

### C3: Missing Command Documentation
**Impact:** Users don't know about train/data/models commands
**Files:** README.md, docs/cli-reference.md
**Action:**
1. Document train command (or mark experimental)
2. Document data command (or mark experimental)
3. Document models command (or mark experimental)
4. Or add note in README: "Advanced commands (train, data, models) documented separately"

## High-Impact Improvements

### H1: Add Structure Analysis Examples
**Files:** README.md, docs/cli-reference.md
**Action:** Add examples showing structure detection with different detectors

### H2: Docstring Audit
**Scope:** Sample check of key modules
**Finding:** From quick review, docstrings appear present on:
- Public APIs (BPMResult, Section classes)
- Main functions (analyze_bpm, analyze_structure)

**Action:**
1. Verify all public API functions have docstrings
2. Check dataclass field documentation
3. Ensure type hints are complete

### H3: Add Cross-References
**Action:** Improve navigation between docs with more internal links

## Minor Improvements

### M1: Update Examples in README
- Add structure analysis example
- Show multiple analysis types
- Demonstrate --workers flag

### M2: Add Troubleshooting Section
- Common errors (ffmpeg not installed, etc.)
- Debug flags usage
- Performance tuning tips

### M3: Week Summaries Location
**Files:** docs/week1-summary.md, docs/week1-final-summary.md
**Action:** Consider moving to separate directory or archiving (these are historical)

## CLAUDE.md Effectiveness Assessment

**Current State:** No project-level CLAUDE.md
**Global CLAUDE.md:** Contains general preferences (no preambles, parallel agents, etc.)

**Recommendation:**
- Current approach appears sufficient
- Project memory handled via memory-graph-draft.json
- Context injection via hooks provides domain knowledge
- No critical need for project CLAUDE.md unless specific repeated instructions needed

## Implementation Approach

### Phase 1: Critical Fixes (Do First)
1. Run all CLI help commands and capture exact output
2. Update cli-reference.md to match reality
3. Update README.md examples to use correct flags
4. Search and replace madmom → beat_this in docs

### Phase 2: High-Impact
1. Add structure analysis examples
2. Document train/data/models or mark experimental
3. Add docstring audit findings

### Phase 3: Polish
1. Add troubleshooting section
2. Improve cross-references
3. Consider archiving week summaries

## Success Criteria

- [ ] All CLI flags in docs match actual CLI output
- [ ] No references to non-existent flags
- [ ] All commands documented or noted as experimental
- [ ] madmom references updated to beat_this
- [ ] Structure analysis examples added
- [ ] README Quick Start covers major features
- [ ] Agent guide includes ML workflow if applicable

## Estimated Effort

- Critical fixes: ~30 minutes (CLI verification + doc updates)
- High-impact improvements: ~20 minutes
- Minor improvements: ~15 minutes
- Total: ~65 minutes

## Notes for Implementation

- Use actual CLI output as source of truth
- Grep codebase to verify flag existence before documenting
- Test examples before adding to docs
- Consider user journey: first-time user → power user → contributor

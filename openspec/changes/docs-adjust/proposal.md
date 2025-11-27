# Proposal: Align Documentation with Implementation

## Problem

Documentation diverges from actual implementation in several critical areas:

1. **Python Version Requirement**: README states 3.12+ but pyproject.toml allows 3.10+
2. **Code Style Configuration**: Python style guide documents 88-char line length but code uses 100
3. **Default Log Level**: CLI reference documents INFO as default but code defaults to WARNING
4. **Library References**: Architecture and development docs heavily reference madmom library, but codebase uses beat_this as primary BPM detector
5. **Unimplemented Features**: Documentation presents placeholder/stub features (structure analysis, Beatport, TuneBat, TOML config) as working

## Solution

Update documentation and configuration to accurately reflect current implementation state:

1. Set Python version requirement to 3.12+ across documentation and pyproject.toml
2. Update python-style.md to document 100-character line length
3. Update cli-reference.md to document WARNING as default log level
4. Replace madmom references with beat_this in architecture and development docs
5. Add clarity about unimplemented features (structure, Beatport, TuneBat, TOML config) with explicit disclaimers

## Scope

**Affected Areas:**
- Configuration: pyproject.toml (Python version, test config)
- Documentation: README.md, docs/python-style.md, docs/cli-reference.md, docs/architecture.md, docs/development.md
- Code comments: Structure analysis, Beatport, TuneBat implementations to clarify they're stubs

## Implementation Strategy

1. **Configuration Changes**: Update requires-python and related classifiers in pyproject.toml
2. **Documentation Updates**: Systematically update each doc file with accurate information
3. **Code Clarity**: Add explicit docstring notes to stub implementations

## Timeline Dependencies

None - all changes are documentation/configuration focused and can be applied independently.

## Risk Assessment

**Low Risk**:
- Documentation-only changes won't affect runtime behavior
- Python version bump to 3.12 is tested by CI
- Changes improve accuracy without breaking functionality

**Considerations**:
- Users on Python 3.10/3.11 will no longer be able to install from updated pyproject.toml
- Documentation changes are non-breaking and improve clarity

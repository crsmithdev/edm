# Design: Core Architecture (Library + CLI)

## Context
This is the foundational architecture for the EDM analysis system. The project is currently in early planning stages with no existing implementation. We need to establish a clean separation between:
- **Core library**: Reusable Python package with all business logic
- **CLI**: User-facing command-line interface

This architecture enables future expansion to other interfaces (web API, desktop GUI, plugin systems) without duplicating logic.

## Goals / Non-Goals

**Goals:**
- Clear separation of concerns: library handles logic, CLI handles I/O
- Modular design that supports independent development of components
- Testable architecture with dependency injection
- Python package structure that follows best practices
- Support for both programmatic and command-line usage

**Non-Goals:**
- Web API or GUI (future capabilities)
- Complete implementation of all analysis features (this change establishes structure only)
- Plugin system or extensibility framework (premature at this stage)
- Multi-language support (Python-first approach)

## Decisions

### Decision 1: Library-First Architecture
**What:** Core library (`edm` package) is completely independent of CLI. All functionality exposed through library API.

**Why:**
- Enables reuse across different interfaces (CLI, API, GUI)
- Makes testing easier (test library without I/O concerns)
- Allows users to import library directly in Python scripts
- Follows Unix philosophy: do one thing well

**Alternatives considered:**
- Monolithic CLI-only application: Rejected because it prevents reuse
- Mixed approach with some logic in CLI: Rejected because it creates tight coupling

### Decision 2: Module Organization by Functional Domain
**What:** Library organized into functional modules:
- `analysis/` - Track analysis algorithms
- `io/` - File reading/writing
- `external/` - API clients for external services
- `models/` - ML model definitions
- `features/` - Audio feature extraction

**Why:**
- Clear ownership and boundaries
- Easy to navigate for new developers
- Supports parallel development
- Aligns with single responsibility principle

**Alternatives considered:**
- Layer-based organization (data/business/presentation): Rejected because cross-cutting concerns in ML projects make this awkward
- Flat structure: Rejected because project will grow too large

### Decision 3: CLI Delegates to Library
**What:** CLI is a thin wrapper that:
1. Parses arguments
2. Calls library functions
3. Formats and displays results

**Why:**
- Keeps CLI simple and maintainable
- All logic testable without CLI
- Easy to add new commands

**Alternatives considered:**
- CLI with embedded logic: Rejected for testability reasons
- Auto-generated CLI from library annotations: Rejected as premature optimization

### Decision 4: Configuration via Files + Environment Variables
**What:** Support both configuration files (YAML/TOML) and environment variables.

**Why:**
- Configuration files for complex settings (model paths, feature extraction parameters)
- Environment variables for secrets and deployment-specific overrides
- Follows 12-factor app principles

**Alternatives considered:**
- Config files only: Rejected because deployment environments need overrides
- Environment variables only: Rejected because complex configs are hard to manage in env vars

## Package Structure

```
edm/
├── src/
│   ├── edm/                    # Core library package
│   │   ├── __init__.py
│   │   ├── analysis/           # Track analysis
│   │   │   ├── __init__.py
│   │   │   ├── bpm.py
│   │   │   ├── structure.py
│   │   │   └── drops.py
│   │   ├── io/                 # File handling
│   │   │   ├── __init__.py
│   │   │   ├── audio.py
│   │   │   └── metadata.py
│   │   ├── external/           # External APIs
│   │   │   ├── __init__.py
│   │   │   ├── spotify.py
│   │   │   ├── beatport.py
│   │   │   └── tunebat.py
│   │   ├── models/             # ML models
│   │   │   ├── __init__.py
│   │   │   └── base.py
│   │   ├── features/           # Feature extraction
│   │   │   ├── __init__.py
│   │   │   ├── spectral.py
│   │   │   └── temporal.py
│   │   └── config.py           # Configuration management
│   └── cli/                    # CLI application
│       ├── __init__.py
│       ├── main.py             # Entry point
│       ├── commands/           # CLI command implementations
│       │   ├── __init__.py
│       │   └── analyze.py
│       ├── output.py           # Rich-based terminal formatting
│       └── profiling.py        # Performance profiling utilities
├── tests/
│   ├── unit/                   # Unit tests
│   │   ├── test_analysis.py
│   │   ├── test_io.py
│   │   └── test_features.py
│   └── integration/            # Integration tests
│       └── test_cli.py
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt            # Pip requirements
└── README.md
```

## API Design Principles

### Library API
- All public functions use type hints
- Return structured data (dataclasses or Pydantic models)
- Raise custom exceptions for error handling
- Accept configuration objects rather than many parameters
- Provide sensible defaults

Example:
```python
from edm.analysis import analyze_track
from edm.config import AnalysisConfig

config = AnalysisConfig(detect_drops=True, extract_bpm=True)
result = analyze_track("track.mp3", config)
print(result.bpm, result.drops)
```

### CLI Design
- Primary command: `edm analyze`
- Analysis selection via `--types` flag: `--types bpm,grid`
- Rich library for formatted output with `--no-color` flag for automation
- Logging for debugging, CLI output for user information
- Timing information always reported

Example:
```bash
edm analyze track.mp3 --types bpm,grid --output results.json
edm analyze track.mp3 --verbose --no-color
edm analyze *.mp3  # Shows timing for all tracks
```

### Logging vs CLI Output

**CLI Output (to stdout/stderr via Rich):**
- Input files being processed
- Progress indicators and status
- Final results (BPM, grid info, etc.)
- Timing information (per-track and summary)
- User-facing error messages

**Logs (to file/stderr via logging module):**
- Detailed computation steps
- Intermediate values and diagnostics
- Feature extraction details
- Model inference internals
- Debug-level troubleshooting info

**Never use print() statements** - all output goes through either Rich (for CLI) or logging (for diagnostics).

## Dependency Strategy

**Core Dependencies:**
- `librosa` - Audio processing
- `madmom` - BPM detection
- `essentia` - EDM-specific features
- `pydantic` - Configuration validation
- `typer` - CLI framework (type-based, modern)
- `rich` - Terminal output formatting
- `spotipy` - Spotify API client
- `requests` - HTTP client for Beatport/TuneBat APIs

**Development Dependencies:**
- `pytest` - Testing
- `black` - Formatting
- `ruff` - Linting
- `mypy` - Type checking

## Risks / Trade-offs

**Risk:** Library API may change frequently in early development
- **Mitigation:** Use semantic versioning, mark APIs as experimental, maintain changelog

**Risk:** CLI and library versions may drift out of sync
- **Mitigation:** Package both together initially, version lock CLI to library

**Trade-off:** More boilerplate code for separation
- **Benefit:** Pays off as project grows and multiple interfaces are added

**Trade-off:** Initial overhead in designing APIs
- **Benefit:** Prevents refactoring pain later when codebase is larger

**Trade-off:** Profiling overhead on every run
- **Benefit:** Early visibility into performance, helps optimize from the start

## Migration Plan

N/A - This is the initial implementation, no migration needed.

## Open Questions

1. **Configuration format:** YAML vs TOML vs JSON?
   - Recommend: TOML (Python native support in 3.11+, clean syntax)

2. **Package distribution:** Single package or separate packages?
   - Recommend: Single package initially, split if needed later

3. **Profiling granularity:** How detailed should timing breakdowns be?
   - Recommend: Start with high-level (load, analyze, export), add detail as needed

4. **Log file location:** Where to write logs by default?
   - Recommend: `~/.local/share/edm/logs/` on Linux/Mac, `%APPDATA%/edm/logs/` on Windows

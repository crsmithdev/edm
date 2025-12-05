# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Data management layer (`src/edm/data/`)
  - Rekordbox XML integration
  - Format converters (JAMS)
  - Data validation utilities
- ML model infrastructure (`src/edm/models/`)
  - CNN backbone architectures
  - Task-specific heads
  - Multitask training support

### Changed
- Updated documentation to reflect actual test structure
- Expanded testing guide with comprehensive examples

## [0.1.0] - 2024-11-01

### Added
- BPM detection with cascading strategy (metadata → beat_this → librosa)
- Structure analysis with MSAF and energy-based detectors
- Bar calculation utilities (time ↔ bar conversion)
- Beat grid generation
- Audio caching with LRU eviction
- Evaluation framework with MAE, RMSE, precision/recall metrics
- CLI with `analyze` and `evaluate` commands
- Structured logging with structlog
- Pydantic-based configuration

### Technical
- Two-tier architecture separating public APIs from detector implementations
- Protocol-based detector interface for extensibility
- Parallel processing support for batch analysis

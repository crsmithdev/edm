# Change: Add Core Architecture (Library + CLI)

## Why
The project needs a foundational architecture to support EDM track analysis. Currently, there's no implementation. This change establishes the separation between business logic (library) and user interface (CLI), enabling modular development and future integration with other interfaces (web API, GUI, etc.).

## What Changes
- Add **core-library** capability: Python library for audio analysis, file handling, and external data retrieval
- Add **cli** capability: Command-line interface for user interaction that delegates to the core library
- Establish clean separation of concerns: library handles all logic, CLI handles I/O and presentation
- Set up package structure for both components

## Impact
- Affected specs: None (new capabilities being added)
- Affected code: New Python package structure
  - `src/edm/` - Core library package
  - `src/cli/` - CLI package
- Establishes foundation for all future development

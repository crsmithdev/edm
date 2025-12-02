# CLI Patterns

Typer conventions and Rich output formatting specific to the EDM project.

## Typer Command Structure

### Main App Definition

Entry point: `src/cli/main.py`

```python
import typer
from rich.console import Console

app = typer.Typer(
    name="edm",
    help="EDM track analysis CLI",
    add_completion=False,  # Disable shell completion
)

# Add subcommands
app.command(name="evaluate")(evaluate_command)
app.add_typer(profile_app, name="profile")

console = Console()
```

**Pattern**:
- Single Typer app instance in `main.py`
- Commands defined in `src/cli/commands/*.py`
- Imported and registered via decorator
- Rich Console for styled output

### Command Registration

Commands live in separate files and are registered in `main.py`:

```python
# src/cli/commands/analyze.py
def analyze_command(
    files: list[Path] = typer.Argument(..., help="Audio files"),
    format: str = typer.Option("table", "--format", "-f"),
):
    # Implementation
    ...

# src/cli/main.py
from cli.commands.analyze import analyze_command

@app.command()
def analyze(
    files: list[Path] = typer.Argument(..., help="Audio files", exists=True),
    format: str = typer.Option("table", "--format", "-f", help="Output format"),
):
    """Analyze audio files for BPM, structure, and bars."""
    console = Console()
    analyze_command(files, format, console)
```

**Pattern**:
- Thin wrapper in `main.py` (docstring, argument validation)
- Heavy logic in `commands/*.py` (easier testing)
- Pass `Console` instance to command functions

## Argument and Option Patterns

### Arguments (Positional)

```python
files: list[Path] = typer.Argument(
    ...,                          # Required (use default value for optional)
    help="Audio files to analyze",
    exists=True,                  # Path validation (must exist)
)
```

**Conventions**:
- Use `list[Path]` for multiple files
- `exists=True` for file/directory inputs
- `...` for required, default value for optional

### Options (Flags)

```python
# Boolean flag
offline: bool = typer.Option(
    False,
    "--offline",
    help="Skip network lookups",
)

# String option with choices (use Enum)
from enum import Enum

class OutputFormat(str, Enum):
    TABLE = "table"
    JSON = "json"
    YAML = "yaml"

format: OutputFormat = typer.Option(
    OutputFormat.TABLE,
    "--format", "-f",
    help="Output format",
)

# Integer option with validation
workers: int = typer.Option(
    1,
    "--workers", "-w",
    min=1, max=16,
    help="Parallel workers",
)
```

**Conventions**:
- Long form: `--flag-name` (kebab-case)
- Short form: `-f` (single letter)
- Always provide help text
- Use Enum for fixed choices (not magic strings)

### Config File Pattern

```python
config_path: Path | None = typer.Option(
    None,
    "--config", "-c",
    help="Configuration file path",
    exists=True,
)
```

Then load via `edm.config.load_config(config_path)`.

## Rich Output Formatting

### Tables

Used for structured multi-record output:

```python
from rich.table import Table
from rich.console import Console

console = Console()
table = Table(title="Analysis Results")
table.add_column("File", style="cyan")
table.add_column("BPM", justify="right")
table.add_column("Duration", justify="right")

for result in results:
    table.add_row(
        Path(result["file"]).name,
        str(result.get("bpm", "N/A")),
        f"{result.get('duration', 'N/A')}s",
    )

console.print(table)
```

**Conventions**:
- Use `style="cyan"` for primary column (filenames)
- `justify="right"` for numbers
- Convert all values to strings
- Use "N/A" for missing data (not empty string)

### JSON Output

```python
import json

json_str = json.dumps(results, indent=2)

if output_path:
    output_path.write_text(json_str)
    console.print(f"Results written to {output_path}")
else:
    console.print(json_str)
```

**Conventions**:
- Always use `indent=2` for readability
- Write to file if `--output` specified, otherwise stdout
- Confirm write success when outputting to file

### YAML Output

```python
import yaml

# Single document
yaml_str = yaml.dump(
    data,
    default_flow_style=None,  # Block style (not inline)
    allow_unicode=True,
    sort_keys=False,           # Preserve insertion order
)

# Multi-document (for batch results)
yaml_str = yaml.dump_all(
    results,  # List of dicts
    default_flow_style=None,
    allow_unicode=True,
    sort_keys=False,
)

console.print(yaml_str)
```

**Conventions**:
- `default_flow_style=None`: Use block style (readable)
- `allow_unicode=True`: Support UTF-8 characters
- `sort_keys=False`: Preserve field order (Pydantic models)
- Use `dump_all()` for batch results (YAML documents separated by `---`)

### Progress Bars

For long-running operations:

```python
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    console=console,
    disable=quiet,  # Disable if --quiet flag set
) as progress:
    task = progress.add_task("Analyzing...", total=len(files))

    for file in files:
        # Process file
        progress.update(task, advance=1)
```

**Conventions**:
- Always include `SpinnerColumn()` (visual feedback)
- `TextColumn` with description (e.g., "Analyzing (4 workers)...")
- `disable=quiet` to respect `--quiet` flag
- Update in small increments (not just at 100%)

### Colored Messages

```python
# Success
console.print("[green]Analysis complete[/green]")

# Warning
console.print("[yellow]No audio files found[/yellow]")

# Error
console.print("[red]Error analyzing file:[/red] {error_message}")

# Info (no color)
console.print("Found 10 files to analyze")
```

**Conventions**:
- Green: Success, completion
- Yellow: Warnings, non-fatal issues
- Red: Errors, failures
- No color: Informational messages

## Error Handling

### Exception Mapping

```python
from edm.exceptions import AnalysisError, AudioFileError

try:
    result = analyze_bpm(filepath)
except AudioFileError as e:
    console.print(f"[red]Could not load {filepath.name}:[/red] {e}")
    return None
except AnalysisError as e:
    console.print(f"[red]Analysis failed for {filepath.name}:[/red] {e}")
    return None
except Exception as e:
    console.print(f"[red]Unexpected error:[/red] {e}")
    logger.exception("unexpected error during analysis")
    raise
```

**Pattern**:
- Catch known exceptions (`AudioFileError`, `AnalysisError`)
- Display user-friendly messages via Rich console
- Log full exception details via structlog
- Re-raise unexpected exceptions (don't swallow)

### Graceful Degradation

```python
# Allow partial results
result = {
    "file": str(filepath),
    "bpm": bpm if bpm else None,
    "structure": sections if sections else None,
}

# Report what failed
if result.get("bpm") is None:
    console.print(f"[yellow]Could not detect BPM for {filepath.name}[/yellow]")
```

**Pattern**:
- Return partial results (don't fail entirely)
- Use `None` for missing fields
- Warn user about missing data (yellow)

## Parallel Processing

### ParallelProcessor Pattern

```python
from edm.processing.parallel import ParallelProcessor

def _worker_function(args: tuple) -> dict:
    """Worker must be top-level function for pickling."""
    filepath, option1, option2 = args

    # Convert strings back to Path if needed
    if isinstance(filepath, str):
        filepath = Path(filepath)

    # Process and return dict
    return {"file": str(filepath), "result": ...}

# Prepare args (must be picklable)
args_list = [
    (str(filepath), option1, option2)  # Convert Path to str
    for filepath in files
]

# Process in parallel
processor = ParallelProcessor(
    worker_fn=_worker_function,
    workers=workers,
    progress_callback=lambda count: progress.update(task, advance=1),
)

results = processor.process(args_list)
```

**Patterns**:
- Worker function must be top-level (not nested, for pickling)
- Convert `Path` to `str` in args (Path not always picklable)
- Convert back to `Path` in worker
- Use progress callback for Rich progress bar updates
- `workers=1` is sequential (no multiprocessing overhead)

### Progress Callback

```python
completed = [0]  # Use list for mutable counter

def progress_callback(count: int):
    advance = count - completed[0]
    if advance > 0:
        progress.update(task, advance=advance)
        completed[0] = count

processor = ParallelProcessor(
    worker_fn=worker,
    workers=workers,
    progress_callback=progress_callback,
)
```

**Pattern**:
- Callback receives total completed count
- Calculate delta from last update
- Update Rich progress bar incrementally
- Use list wrapper for mutable state in closure

## Output Format Selection

### Common Pattern

```python
if format == "yaml":
    output_yaml(results, output_path, console, quiet)
elif format == "json" or output_path:
    output_json(results, output_path, console, quiet)
else:
    output_table(results, console, quiet)
```

**Rules**:
- `--format yaml`: YAML output
- `--format json`: JSON output
- `--format table` (default): Rich table (human-readable)
- `--output file.json`: Force JSON regardless of `--format`
- `--quiet`: Suppress tables and progress (JSON/YAML still output)

## Logging Integration

### CLI Logging Setup

```python
from edm.logging import configure_logging

# Configure before any analysis
configure_logging(
    log_level="INFO",      # From --log-level flag
    json_logs=False,       # From --json-logs flag
    log_file=log_path,     # From --log-file flag
)

logger = structlog.get_logger(__name__)
logger.info("starting analysis", file_count=len(files))
```

**Conventions**:
- Configure logging once in command entry point
- Use structured logging: `logger.info("message", key=value)`
- Natural language messages: "analyzing file" not "FILE_ANALYSIS_START"
- Bind context for request tracing: `logger.bind(request_id=uuid4())`

### Log Levels

- `debug`: Algorithm internals (beat intervals, energy values)
- `info`: High-level progress (starting analysis, detected BPM)
- `warning`: Missing data, fallbacks used (metadata unavailable)
- `error`: Failures with exception info

## Testing CLI Commands

### Typer Testing Pattern

```python
from typer.testing import CliRunner
from cli.main import app

runner = CliRunner()

def test_analyze_command():
    result = runner.invoke(app, ["analyze", "test.mp3"])
    assert result.exit_code == 0
    assert "BPM" in result.output
```

**Conventions**:
- Use `CliRunner` from Typer (not Click)
- Test exit codes (0 = success)
- Assert on output strings for user-facing messages
- Use fixtures for temporary files

### Testing Output Formats

```python
def test_json_output(tmp_path):
    output_file = tmp_path / "results.json"
    result = runner.invoke(app, [
        "analyze", "test.mp3",
        "--format", "json",
        "--output", str(output_file)
    ])

    assert result.exit_code == 0
    assert output_file.exists()

    data = json.loads(output_file.read_text())
    assert len(data) == 1
    assert data[0]["file"] == "test.mp3"
```

## Command Composition

### Subcommands

```python
# Create sub-app
profile_app = typer.Typer(help="Profiling commands")

@profile_app.command()
def cpu():
    """Profile CPU usage."""
    ...

@profile_app.command()
def memory():
    """Profile memory usage."""
    ...

# Register in main app
app.add_typer(profile_app, name="profile")
```

**Usage**:
```bash
edm profile cpu track.mp3
edm profile memory track.mp3
```

**Conventions**:
- Group related commands under sub-apps
- Use clear command names (verbs: analyze, evaluate, profile)
- Provide help text for discoverability

## Common Patterns Summary

| Pattern | Usage | Example |
|---------|-------|---------|
| `list[Path]` arguments | Multiple file inputs | `files: list[Path] = typer.Argument(...)` |
| Enum for choices | Fixed option values | `OutputFormat.JSON` |
| Rich tables | Human-readable results | `Table(title="Results")` |
| Progress bars | Long operations | `Progress(SpinnerColumn(), BarColumn())` |
| JSON output | Machine-readable | `json.dumps(results, indent=2)` |
| YAML multi-doc | Batch results | `yaml.dump_all(results)` |
| Colored messages | Status/errors | `[green]Success[/green]` |
| Parallel processing | Multi-file analysis | `ParallelProcessor(worker_fn, workers=4)` |
| Config file option | User preferences | `typer.Option(None, "--config")` |
| Quiet flag | Scripting | `typer.Option(False, "--quiet")` |

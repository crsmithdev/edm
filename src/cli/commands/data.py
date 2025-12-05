"""Data management CLI commands."""

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from edm.data.export import (
    export_to_csv,
    export_to_json,
    export_to_pytorch,
    export_to_tensorflow,
)
from edm.data.metadata import AnnotationTier
from edm.data.schema import Annotation

app = typer.Typer(help="Data management commands for annotations and datasets")
console = Console()


@app.command()
def stats(
    data_dir: Path = typer.Option(
        Path("data/annotations"), "--data-dir", "-d", help="Directory containing annotation files"
    ),
) -> None:
    """Display dataset statistics.

    Shows overview of annotation counts, tiers, confidence scores,
    and validation status.
    """
    if not data_dir.exists():
        rprint(f"[red]Error:[/red] Data directory not found: {data_dir}")
        raise typer.Exit(1)

    # Load all annotations
    yaml_files = list(data_dir.glob("*.yaml"))
    if not yaml_files:
        rprint(f"[yellow]Warning:[/yellow] No annotation files found in {data_dir}")
        return

    annotations: list[Annotation] = []
    failed_files: list[str] = []

    for yaml_file in yaml_files:
        try:
            ann = Annotation.from_yaml(yaml_file)
            annotations.append(ann)
        except Exception as e:
            failed_files.append(f"{yaml_file.name}: {e}")

    if not annotations:
        rprint("[red]Error:[/red] No valid annotations found")
        if failed_files:
            rprint("\n[yellow]Failed to load:[/yellow]")
            for error in failed_files:
                rprint(f"  - {error}")
        raise typer.Exit(1)

    # Calculate statistics
    total = len(annotations)
    tier_counts = {
        AnnotationTier.VERIFIED: 0,
        AnnotationTier.AUTO_CLEANED: 0,
        AnnotationTier.AUTO_GENERATED: 0,
    }
    confidences: list[float] = []
    needs_review = 0

    for ann in annotations:
        tier_counts[ann.metadata.tier] += 1
        confidences.append(ann.metadata.confidence)
        if "needs_review" in ann.metadata.flags:
            needs_review += 1

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Display statistics
    rprint("\n[bold cyan]Dataset Statistics[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total tracks", str(total))
    table.add_row(
        "Tier 1 (verified)",
        f"{tier_counts[AnnotationTier.VERIFIED]} "
        f"({tier_counts[AnnotationTier.VERIFIED] / total * 100:.1f}%)",
    )
    table.add_row(
        "Tier 2 (auto-cleaned)",
        f"{tier_counts[AnnotationTier.AUTO_CLEANED]} "
        f"({tier_counts[AnnotationTier.AUTO_CLEANED] / total * 100:.1f}%)",
    )
    table.add_row(
        "Tier 3 (auto-generated)",
        f"{tier_counts[AnnotationTier.AUTO_GENERATED]} "
        f"({tier_counts[AnnotationTier.AUTO_GENERATED] / total * 100:.1f}%)",
    )
    table.add_row("Average confidence", f"{avg_confidence:.2f}")
    table.add_row("Needs review", str(needs_review))

    console.print(table)

    if failed_files:
        rprint("\n[yellow]Failed to load:[/yellow]")
        for error in failed_files[:10]:  # Show first 10
            rprint(f"  - {error}")
        if len(failed_files) > 10:
            rprint(f"  ... and {len(failed_files) - 10} more")


@app.command()
def validate(
    data_dir: Path = typer.Option(
        Path("data/annotations"), "--data-dir", "-d", help="Directory containing annotation files"
    ),
) -> None:
    """Validate all annotations against schema.

    Checks for schema violations and structural issues like:
    - Invalid section labels
    - Bars/times not in order
    - Sections too short (< 4 bars)
    """
    if not data_dir.exists():
        rprint(f"[red]Error:[/red] Data directory not found: {data_dir}")
        raise typer.Exit(1)

    yaml_files = list(data_dir.glob("*.yaml"))
    if not yaml_files:
        rprint(f"[yellow]Warning:[/yellow] No annotation files found in {data_dir}")
        return

    valid_count = 0
    invalid_files: list[tuple[str, list[str]]] = []

    for yaml_file in yaml_files:
        try:
            ann = Annotation.from_yaml(yaml_file)
            warnings = ann.validate_structure()
            if warnings:
                invalid_files.append((yaml_file.name, warnings))
            else:
                valid_count += 1
        except Exception as e:
            invalid_files.append((yaml_file.name, [f"Schema error: {e}"]))

    # Display results
    total = len(yaml_files)
    rprint("\n[bold cyan]Validation Results[/bold cyan]\n")

    if valid_count == total:
        rprint(f"[green]✓[/green] All {total} annotations are valid")
        return

    rprint(f"[green]✓[/green] {valid_count} annotations valid")
    rprint(f"[red]✗[/red] {len(invalid_files)} annotations have issues:\n")

    for filename, warnings in invalid_files[:20]:  # Show first 20
        rprint(f"  [yellow]{filename}[/yellow]")
        for warning in warnings:
            rprint(f"    - {warning}")

    if len(invalid_files) > 20:
        rprint(f"\n  ... and {len(invalid_files) - 20} more files with issues")

    raise typer.Exit(1)


@app.command()
def export(
    format: str = typer.Argument(
        ..., help="Export format: json, pytorch, tensorflow, csv-time, csv-bar"
    ),
    output: Path = typer.Argument(..., help="Output file path"),
    data_dir: Path = typer.Option(
        Path("data/annotations"), "--data-dir", "-d", help="Directory containing annotation files"
    ),
) -> None:
    """Export annotations to ML framework format.

    Supported formats:
    - json: Plain JSON format
    - pytorch: PyTorch Dataset-compatible format
    - tensorflow: TensorFlow Dataset-compatible format
    - csv-time: CSV with time-based sections
    - csv-bar: CSV with bar-based sections
    """
    if not data_dir.exists():
        rprint(f"[red]Error:[/red] Data directory not found: {data_dir}")
        raise typer.Exit(1)

    # Load all valid annotations
    yaml_files = list(data_dir.glob("*.yaml"))
    annotations: list[Annotation] = []

    for yaml_file in yaml_files:
        try:
            ann = Annotation.from_yaml(yaml_file)
            annotations.append(ann)
        except Exception as e:
            rprint(f"[yellow]Warning:[/yellow] Skipping {yaml_file.name}: {e}")

    if not annotations:
        rprint("[red]Error:[/red] No valid annotations to export")
        raise typer.Exit(1)

    # Export based on format
    try:
        if format == "json":
            export_to_json(annotations, output)
        elif format == "pytorch":
            export_to_pytorch(annotations, output)
        elif format == "tensorflow":
            export_to_tensorflow(annotations, output)
        elif format == "csv-time":
            export_to_csv(annotations, output, format="time")
        elif format == "csv-bar":
            export_to_csv(annotations, output, format="bar")
        else:
            rprint(f"[red]Error:[/red] Unknown format: {format}")
            rprint("Supported: json, pytorch, tensorflow, csv-time, csv-bar")
            raise typer.Exit(1)

        rprint(f"[green]✓[/green] Exported {len(annotations)} annotations to {output}")

    except Exception as e:
        rprint(f"[red]Error:[/red] Export failed: {e}")
        raise typer.Exit(1)


@app.command()
def diff(
    version1: str = typer.Argument(..., help="First version (e.g., 'v1.0' or git commit)"),
    version2: str = typer.Argument(..., help="Second version (e.g., 'v1.1' or 'HEAD')"),
) -> None:
    """Compare dataset versions using DVC.

    Shows differences in annotations between two git commits or tags.
    Requires DVC to be set up and data tracked with DVC.
    """
    try:
        # Use DVC diff to compare versions
        result = subprocess.run(
            ["dvc", "diff", version1, version2, "data.dvc"],
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout:
            rprint(f"\n[bold cyan]DVC Diff: {version1} → {version2}[/bold cyan]\n")
            rprint(result.stdout)
        else:
            rprint(f"[green]No differences found between {version1} and {version2}[/green]")

    except subprocess.CalledProcessError as e:
        rprint(f"[red]Error:[/red] DVC diff failed: {e.stderr}")
        rprint("\nMake sure DVC is initialized and data/ is tracked with DVC")
        raise typer.Exit(1)
    except FileNotFoundError:
        rprint("[red]Error:[/red] DVC not found. Install with: pip install dvc")
        raise typer.Exit(1)


@app.command()
def flag(
    files: list[Path] = typer.Argument(..., help="Annotation files to flag"),
    add_flag: Optional[str] = typer.Option(
        None, "--add", help="Flag to add (e.g., 'needs_review')"
    ),
    remove_flag: Optional[str] = typer.Option(None, "--remove", help="Flag to remove"),
) -> None:
    """Add or remove flags from annotations.

    Common flags: needs_review, boundary_uncertain, low_confidence
    """
    if not add_flag and not remove_flag:
        rprint("[red]Error:[/red] Must specify --add or --remove")
        raise typer.Exit(1)

    modified = 0
    for file_path in files:
        try:
            ann = Annotation.from_yaml(file_path)

            if add_flag and add_flag not in ann.metadata.flags:
                ann.metadata.flags.append(add_flag)
                modified += 1

            if remove_flag and remove_flag in ann.metadata.flags:
                ann.metadata.flags.remove(remove_flag)
                modified += 1

            ann.to_yaml(file_path)

        except Exception as e:
            rprint(f"[yellow]Warning:[/yellow] Failed to process {file_path.name}: {e}")

    if modified > 0:
        rprint(f"[green]✓[/green] Modified {modified} annotations")
    else:
        rprint("[yellow]No changes made[/yellow]")


@app.command()
def tier(
    files: list[Path] = typer.Argument(..., help="Annotation files to update"),
    set_tier: int = typer.Option(..., "--set", help="Tier to set (1, 2, or 3)"),
    verified_by: Optional[str] = typer.Option(
        None, "--verified-by", help="User name for tier 1 (verified) annotations"
    ),
) -> None:
    """Update tier for annotations.

    Tiers:
    - 1: Verified (manually checked, requires --verified-by)
    - 2: Auto-cleaned (refined from source)
    - 3: Auto-generated (needs review)
    """
    if set_tier not in [1, 2, 3]:
        rprint("[red]Error:[/red] Tier must be 1, 2, or 3")
        raise typer.Exit(1)

    if set_tier == 1 and not verified_by:
        rprint("[red]Error:[/red] Tier 1 requires --verified-by <username>")
        raise typer.Exit(1)

    modified = 0
    for file_path in files:
        try:
            ann = Annotation.from_yaml(file_path)
            ann.metadata.tier = AnnotationTier(set_tier)

            if set_tier == 1:
                ann.metadata.verified_by = verified_by
            else:
                ann.metadata.verified_by = None

            ann.to_yaml(file_path)
            modified += 1

        except Exception as e:
            rprint(f"[yellow]Warning:[/yellow] Failed to process {file_path.name}: {e}")

    rprint(f"[green]✓[/green] Updated tier for {modified} annotations")


if __name__ == "__main__":
    app()

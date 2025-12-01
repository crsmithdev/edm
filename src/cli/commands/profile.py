"""CLI commands for profiling."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

profile_app = typer.Typer(
    name="profile",
    help="Performance profiling commands",
)

console = Console()


@profile_app.command("list")
def list_baselines(
    baselines_dir: Path = typer.Option(
        Path("benchmarks/baselines"),
        "--dir",
        "-d",
        help="Baselines directory",
    ),
) -> None:
    """List saved performance baselines."""
    from edm.profiling.baseline import BaselineStore

    store = BaselineStore(baselines_dir)
    baselines = store.list()

    if not baselines:
        console.print("[dim]No baselines found[/dim]")
        return

    table = Table(title="Performance Baselines")
    table.add_column("Name", style="cyan")
    table.add_column("Commit", style="dim")
    table.add_column("Branch", style="green")
    table.add_column("Profiles", justify="right")
    table.add_column("Created", style="dim")

    for name in sorted(baselines):
        baseline = store.load(name)
        if baseline:
            table.add_row(
                name,
                baseline.metadata.commit[:7],
                baseline.metadata.branch,
                str(len(baseline.profiles)),
                baseline.metadata.timestamp.strftime("%Y-%m-%d %H:%M"),
            )

    console.print(table)


@profile_app.command("show")
def show_baseline(
    name: str = typer.Argument(..., help="Baseline name to show"),
    baselines_dir: Path = typer.Option(
        Path("benchmarks/baselines"),
        "--dir",
        "-d",
        help="Baselines directory",
    ),
) -> None:
    """Show details of a specific baseline."""
    from edm.profiling.baseline import BaselineStore

    store = BaselineStore(baselines_dir)
    baseline = store.load(name)

    if baseline is None:
        console.print(f"[red]Baseline '{name}' not found[/red]")
        raise typer.Exit(code=1)

    # Metadata
    console.print(f"\n[bold cyan]{baseline.metadata.name}[/bold cyan]")
    console.print(f"  Commit: {baseline.metadata.commit}")
    console.print(f"  Branch: {baseline.metadata.branch}")
    console.print(f"  Created: {baseline.metadata.timestamp}")

    if baseline.metadata.system_info:
        console.print("  System:")
        for key, value in baseline.metadata.system_info.items():
            console.print(f"    {key}: {value}")

    # Profiles
    if baseline.profiles:
        console.print("\n[bold]Profiles:[/bold]")
        table = Table()
        table.add_column("Name", style="cyan")
        table.add_column("Type")
        table.add_column("Wall Time", justify="right")
        table.add_column("CPU Time", justify="right")
        table.add_column("Peak Memory", justify="right")

        for profile_name, result in baseline.profiles.items():
            table.add_row(
                profile_name,
                result.profile_type,
                f"{result.wall_time:.3f}s",
                f"{result.cpu_time:.3f}s" if result.cpu_time > 0 else "-",
                f"{result.peak_memory_mb:.1f}MB" if result.peak_memory_mb > 0 else "-",
            )

        console.print(table)


@profile_app.command("compare")
def compare_baselines(
    baseline_name: str = typer.Argument(..., help="Baseline to compare against"),
    current_name: str = typer.Argument(..., help="Current baseline to compare"),
    baselines_dir: Path = typer.Option(
        Path("benchmarks/baselines"),
        "--dir",
        "-d",
        help="Baselines directory",
    ),
    threshold: float = typer.Option(
        20.0,
        "--threshold",
        "-t",
        help="Regression threshold percentage",
    ),
) -> None:
    """Compare two baselines for regressions."""
    from edm.profiling.baseline import BaselineStore, compare_baseline

    store = BaselineStore(baselines_dir)

    baseline = store.load(baseline_name)
    if baseline is None:
        console.print(f"[red]Baseline '{baseline_name}' not found[/red]")
        raise typer.Exit(code=1)

    current = store.load(current_name)
    if current is None:
        console.print(f"[red]Baseline '{current_name}' not found[/red]")
        raise typer.Exit(code=1)

    results = compare_baseline(baseline, current.profiles, threshold / 100)

    if not results:
        console.print("[yellow]No comparable profiles found[/yellow]")
        return

    table = Table(title=f"Comparison: {baseline_name} → {current_name}")
    table.add_column("Profile", style="cyan")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("Status")

    has_regression = False
    for result in results:
        diff_style = "red" if result.is_regression else "green"
        status = "[red]REGRESSION[/red]" if result.is_regression else "[green]OK[/green]"

        if result.is_regression:
            has_regression = True

        # Format values based on metric type
        if result.metric == "peak_memory_mb":
            baseline_fmt = f"{result.baseline_value:.1f}MB"
            current_fmt = f"{result.current_value:.1f}MB"
        else:
            baseline_fmt = f"{result.baseline_value:.3f}s"
            current_fmt = f"{result.current_value:.3f}s"

        table.add_row(
            result.profile_name,
            result.metric,
            baseline_fmt,
            current_fmt,
            f"[{diff_style}]{result.diff_percent:+.1f}%[/{diff_style}]",
            status,
        )

    console.print(table)

    if has_regression:
        console.print(f"\n[red]Performance regressions detected (threshold: {threshold}%)[/red]")
        raise typer.Exit(code=1)
    else:
        console.print("\n[green]No regressions detected[/green]")


@profile_app.command("delete")
def delete_baseline(
    name: str = typer.Argument(..., help="Baseline name to delete"),
    baselines_dir: Path = typer.Option(
        Path("benchmarks/baselines"),
        "--dir",
        "-d",
        help="Baselines directory",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete a baseline."""
    from edm.profiling.baseline import BaselineStore

    store = BaselineStore(baselines_dir)

    if not force:
        confirm = typer.confirm(f"Delete baseline '{name}'?")
        if not confirm:
            raise typer.Abort()

    if store.delete(name):
        console.print(f"[green]Deleted baseline '{name}'[/green]")
    else:
        console.print(f"[red]Baseline '{name}' not found[/red]")
        raise typer.Exit(code=1)

"""CLI commands for MLflow model management."""

import typer
from rich.console import Console
from rich.table import Table

from edm.registry.mlflow_registry import ModelRegistry

app = typer.Typer(help="Manage MLflow model registry")
console = Console()


@app.command("list")
def list_models(
    max_results: int = typer.Option(10, help="Maximum number of models to display"),
) -> None:
    """List registered models with metadata."""
    registry = ModelRegistry()
    models = registry.list_models(max_results=max_results)

    if not models:
        console.print("[yellow]No models found in registry[/yellow]")
        return

    # Create table
    table = Table(title="Registered Models")
    table.add_column("Version", justify="right", style="cyan")
    table.add_column("Stage", style="green")
    table.add_column("Run Name", style="blue")
    table.add_column("Val Loss", justify="right")
    table.add_column("Backbone")
    table.add_column("Created")

    # Add rows
    for model in models:
        import datetime

        created = datetime.datetime.fromtimestamp(model["created_at"] / 1000).strftime(
            "%Y-%m-%d %H:%M"
        )
        val_loss = f"{model['metrics'].get('best_val_loss', 0):.4f}"
        backbone = model["params"].get("backbone", "unknown")

        table.add_row(
            str(model["version"]),
            model["stage"],
            model["run_name"],
            val_loss,
            backbone,
            created,
        )

    console.print(table)


@app.command("info")
def model_info(
    version: int = typer.Argument(..., help="Model version number"),
) -> None:
    """Show detailed information for a specific model version."""
    registry = ModelRegistry()
    info = registry.get_model_info(version)

    if not info:
        console.print(f"[red]Model version {version} not found[/red]")
        raise typer.Exit(1)

    # Print details
    console.print(f"\n[bold]Model Version {info['version']}[/bold]")
    console.print(f"Stage: [green]{info['stage']}[/green]")
    console.print(f"Run: {info['run_name']}")
    console.print(f"Run ID: {info['run_id']}")

    # Metrics
    console.print("\n[bold]Metrics:[/bold]")
    for key, value in info["metrics"].items():
        console.print(f"  {key}: {value:.4f}")

    # Parameters
    console.print("\n[bold]Parameters:[/bold]")
    for key, value in info["params"].items():
        console.print(f"  {key}: {value}")

    # Tags
    if info["tags"]:
        console.print("\n[bold]Tags:[/bold]")
        for key, value in info["tags"].items():
            if not key.startswith("mlflow."):
                console.print(f"  {key}: {value}")


@app.command("promote")
def promote_model(
    version: int = typer.Argument(..., help="Model version number"),
    stage: str = typer.Argument(..., help="Target stage (Staging/Production/Archived)"),
) -> None:
    """Promote model to a specific stage."""
    valid_stages = ["Staging", "Production", "Archived"]

    if stage not in valid_stages:
        console.print(f"[red]Invalid stage. Must be one of: {', '.join(valid_stages)}[/red]")
        raise typer.Exit(1)

    registry = ModelRegistry()

    # Get model info first
    info = registry.get_model_info(version)
    if not info:
        console.print(f"[red]Model version {version} not found[/red]")
        raise typer.Exit(1)

    # Confirm promotion
    console.print("\nPromoting model:")
    console.print(f"  Version: {version}")
    console.print(f"  Run: {info['run_name']}")
    console.print(f"  Current stage: {info['stage']}")
    console.print(f"  Target stage: [green]{stage}[/green]")

    confirm = typer.confirm("\nProceed?")
    if not confirm:
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)

    # Promote
    registry.promote_model(version, stage)
    console.print(f"[green]✓ Model version {version} promoted to {stage}[/green]")


@app.command("load-production")
def load_production() -> None:
    """Show path to current production model."""
    registry = ModelRegistry()
    model_path = registry.load_production_model()

    if not model_path:
        console.print("[yellow]No production model found[/yellow]")
        raise typer.Exit(1)

    console.print(f"Production model: {model_path}")

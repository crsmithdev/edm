"""Evaluate command - accuracy evaluation for analysis algorithms."""

from pathlib import Path

import typer

from edm.evaluation import evaluate_bpm

evaluate_app = typer.Typer(help="Evaluate accuracy of analysis algorithms")


@evaluate_app.command("bpm")
def evaluate_bpm_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Directory containing audio files to analyze",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    reference: str = typer.Option(
        ...,
        "--reference",
        help="Reference source: 'spotify', 'metadata', or path to CSV/JSON file",
    ),
    sample_size: int = typer.Option(
        100,
        "--sample-size",
        help="Number of files to sample (ignored if --full)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output directory (default: benchmarks/results/accuracy/bpm/)",
        file_okay=False,
        dir_okay=True,
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible sampling",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Evaluate all files (ignore --sample-size)",
    ),
    tolerance: float = typer.Option(
        2.5,
        "--tolerance",
        help="BPM tolerance for accuracy calculation",
    ),
):
    """Evaluate BPM detection accuracy.

    Examples:

        edm evaluate bpm --source ~/music --reference tests/fixtures/reference/bpm_tagged.csv

        edm evaluate bpm --source ~/music --reference spotify

        edm evaluate bpm --source ~/music --reference metadata

        edm evaluate bpm --source ~/music --reference metadata --full --seed 42
    """
    try:
        evaluate_bpm(
            source_path=source,
            reference_source=reference,
            sample_size=sample_size,
            output_dir=output,
            seed=seed,
            full=full,
            tolerance=tolerance,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

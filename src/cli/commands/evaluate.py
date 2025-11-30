"""Evaluate command - accuracy evaluation for analysis algorithms."""

from pathlib import Path

import typer

from edm.evaluation import evaluate_bpm, evaluate_structure

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
        help="Output directory (default: data/accuracy/bpm/)",
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
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel workers for evaluation (default: 1 = sequential)",
    ),
):
    """Evaluate BPM detection accuracy.

    Examples:

        edm evaluate bpm --source ~/music --reference data/annotations/bpm_tagged.csv

        edm evaluate bpm --source ~/music --reference spotify

        edm evaluate bpm --source ~/music --reference metadata

        edm evaluate bpm --source ~/music --reference metadata --full --seed 42

        edm evaluate bpm --source ~/music --reference metadata --workers 4  # Parallel
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
            workers=workers,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@evaluate_app.command("structure")
def evaluate_structure_command(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Directory containing audio files to analyze",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    reference: Path = typer.Option(
        ...,
        "--reference",
        help="Path to CSV file with ground truth annotations (filename,start,end,label)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    sample_size: int = typer.Option(
        100,
        "--sample-size",
        help="Number of files to sample (ignored if --full)",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output directory (default: data/accuracy/structure/)",
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
        2.0,
        "--tolerance",
        help="Boundary tolerance in seconds for section matching",
    ),
    detector: str = typer.Option(
        "auto",
        "--detector",
        help="Structure detector to use: auto (default), msaf, or energy",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel workers for evaluation (default: 1 = sequential)",
    ),
):
    """Evaluate structure detection accuracy.

    Reference CSV format:
        filename,start,end,label

    Where label is one of: intro, buildup, drop, breakdown, outro

    Examples:

        edm evaluate structure --source ~/music --reference annotations.csv

        edm evaluate structure --source ~/music --reference annotations.csv --full

        edm evaluate structure --source ~/music --reference annotations.csv --tolerance 3.0

        edm evaluate structure --source ~/music --reference annotations.csv --detector energy
    """
    # Note: workers parameter is accepted but not used for structure evaluation
    # Structure evaluation is sequential to ensure deterministic results
    _ = workers  # Unused

    try:
        evaluate_structure(
            source_path=source,
            reference_path=reference,
            sample_size=sample_size,
            output_dir=output,
            seed=seed,
            full=full,
            tolerance=tolerance,
            detector=detector,
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

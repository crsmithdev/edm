"""Training command for CLI."""

from pathlib import Path

import typer
from rich.console import Console

from edm.models.multitask import create_model
from edm.training import create_dataloaders
from edm.training.losses import MultiTaskLoss
from edm.training.trainer import Trainer, TrainingConfig

console = Console()


def train_command(
    annotation_dir: Path = typer.Argument(..., help="Directory with YAML annotations", exists=True),
    audio_dir: Path | None = typer.Option(
        None, "--audio-dir", "-a", help="Directory with audio files"
    ),
    output_dir: Path = typer.Option(
        Path("outputs/training"), "--output", "-o", help="Base output directory"
    ),
    run_name: str | None = typer.Option(
        None, "--run-name", "-r", help="Run name (auto-generated if not provided)"
    ),
    backbone: str = typer.Option(
        "mert-95m", "--backbone", "-b", help="Backbone type (mert-95m, mert-330m, cnn)"
    ),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size"),
    num_epochs: int = typer.Option(50, "--epochs", "-e", help="Number of epochs"),
    learning_rate: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    train_split: float = typer.Option(0.8, "--train-split", help="Training split fraction"),
    duration: float | None = typer.Option(
        30.0, "--duration", "-d", help="Audio segment duration (None for full tracks)"
    ),
    num_workers: int = typer.Option(4, "--workers", "-w", help="Number of dataloader workers"),
    boundary_weight: float = typer.Option(
        1.0, "--boundary-weight", help="Boundary detection loss weight"
    ),
    energy_weight: float = typer.Option(
        0.5, "--energy-weight", help="Energy prediction loss weight"
    ),
    beat_weight: float = typer.Option(0.5, "--beat-weight", help="Beat detection loss weight"),
    label_weight: float = typer.Option(
        0.3, "--label-weight", help="Label classification loss weight"
    ),
    enable_boundary: bool = typer.Option(
        True, "--boundary/--no-boundary", help="Enable boundary detection"
    ),
    enable_energy: bool = typer.Option(
        True, "--energy/--no-energy", help="Enable energy prediction"
    ),
    enable_beat: bool = typer.Option(True, "--beat/--no-beat", help="Enable beat detection"),
    enable_label: bool = typer.Option(
        False, "--label/--no-label", help="Enable label classification"
    ),
    resume: Path | None = typer.Option(None, "--resume", help="Resume from checkpoint"),
) -> None:
    """Train EDM structure detection model.

    Args:
        annotation_dir: Directory with YAML annotations
        audio_dir: Directory with audio files (if None, use paths from annotations)
        output_dir: Output directory for checkpoints and logs
        backbone: Backbone type ('mert-95m', 'mert-330m', or 'cnn')
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        train_split: Training split fraction (rest is validation)
        duration: Audio segment duration in seconds (None = full tracks)
        num_workers: Number of dataloader workers
        boundary_weight: Loss weight for boundary detection
        energy_weight: Loss weight for energy prediction
        beat_weight: Loss weight for beat detection
        label_weight: Loss weight for label classification
        enable_boundary: Enable boundary detection head
        enable_energy: Enable energy prediction head
        enable_beat: Enable beat detection head
        enable_label: Enable label classification head
        resume: Path to checkpoint to resume training from
    """
    # Generate run name if not provided
    if run_name is None:
        from edm.training.trainer import generate_run_name

        run_name = generate_run_name(backbone=backbone, description="default")

    # Create run-specific output directory
    run_output_dir = output_dir / run_name

    console.print("[bold cyan]EDM Model Training[/bold cyan]")
    console.print(f"Run name: {run_name}")
    console.print(f"Annotations: {annotation_dir}")
    console.print(f"Audio: {audio_dir or 'from annotations'}")
    console.print(f"Output: {run_output_dir}")
    console.print()

    # Create dataloaders
    console.print("[yellow]Creating dataloaders...[/yellow]")
    train_loader, val_loader = create_dataloaders(
        annotation_dir=annotation_dir,
        audio_dir=audio_dir,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=num_workers,
        duration=duration,
    )
    console.print(f"Training batches: {len(train_loader)}")
    console.print(f"Validation batches: {len(val_loader)}")
    console.print()

    # Get number of classes from dataset
    dataset = train_loader.dataset.dataset  # type: ignore[attr-defined]
    num_classes = dataset.num_classes if enable_label else 6  # default fallback

    # Create model
    console.print(f"[yellow]Creating {backbone} model...[/yellow]")
    model = create_model(
        backbone_type=backbone,  # type: ignore[arg-type]
        enable_boundary=enable_boundary,
        enable_energy=enable_energy,
        enable_beat=enable_beat,
        enable_label=enable_label,
        num_classes=num_classes,
    )

    # Print model info
    param_counts = model.count_parameters()
    console.print(f"Total parameters: {param_counts['total']:,}")
    console.print(f"Trainable parameters: {param_counts['total_trainable']:,}")
    console.print()

    # Create loss function
    loss_fn = MultiTaskLoss(
        boundary_weight=boundary_weight,
        energy_weight=energy_weight,
        beat_weight=beat_weight,
        label_weight=label_weight,
    )

    # Create trainer
    config = TrainingConfig(
        output_dir=run_output_dir,
        run_name=run_name,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_fn=loss_fn,
    )

    # Resume from checkpoint if specified
    if resume:
        console.print(f"[yellow]Resuming from checkpoint: {resume}[/yellow]")
        trainer.load_checkpoint(resume)
        console.print()

    # Start training
    console.print("[bold green]Starting training...[/bold green]")
    console.print()

    try:
        trainer.train()
        console.print()
        console.print("[bold green]Training complete![/bold green]")
        console.print(f"Best model saved to: {run_output_dir / 'checkpoints' / 'best.pt'}")
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Training interrupted by user[/yellow]")
        console.print(f"Latest checkpoint: {run_output_dir}")
    except Exception as e:
        console.print()
        console.print(f"[red]Training failed: {e}[/red]")
        raise

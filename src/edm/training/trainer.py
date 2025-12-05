"""Training infrastructure for EDM structure detection models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from edm.models.multitask import MultiTaskModel
from edm.training.losses import MultiTaskLoss


def generate_run_name(backbone: str, description: str = "default") -> str:
    """Generate timestamped run name.

    Args:
        backbone: Backbone type (mert-95m, mert-330m, cnn, etc.)
        description: Short description for the run

    Returns:
        Run name in format: run_YYYYMMDD_HHMMSS_{backbone}_{description}
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize description
    safe_desc = description.replace(" ", "_").replace("/", "_")
    return f"run_{timestamp}_{backbone}_{safe_desc}"


@dataclass
class TrainingConfig:
    """Training configuration.

    Attributes:
        output_dir: Directory for checkpoints and logs
        run_name: Optional run name for organizing experiments
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        scheduler: LR scheduler type ('cosine', 'onecycle', or None)
        warmup_epochs: Number of warmup epochs
        gradient_clip: Max gradient norm (None = no clipping)
        save_every: Save checkpoint every N epochs
        eval_every: Run evaluation every N epochs
        log_every: Log metrics every N batches
        device: Training device ('cuda' or 'cpu')
    """

    output_dir: Path
    run_name: str | None = None
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str | None = "cosine"
    warmup_epochs: int = 5
    gradient_clip: float | None = 1.0
    save_every: int = 5
    eval_every: int = 1
    log_every: int = 10
    device: str | None = None


class Trainer:
    """Trainer for multi-task EDM structure detection.

    Handles training loop, optimization, checkpointing, and logging.
    """

    def __init__(
        self,
        model: MultiTaskModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        loss_fn: MultiTaskLoss | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Multi-task model to train
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Training configuration
            loss_fn: Loss function (default: MultiTaskLoss with default weights)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup device
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup loss function
        self.loss_fn = loss_fn or MultiTaskLoss()
        self.loss_fn.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup AMP (Automatic Mixed Precision)
        self.use_amp = self.device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Setup logging - create organized output structure
        self.checkpoints_dir = self.config.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.config.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.logs_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler.

        Returns:
            LR scheduler or None
        """
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01,
            )
        elif self.config.scheduler == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=len(self.train_loader),
            )
        else:
            return None

    def train(self) -> None:
        """Run full training loop."""
        from datetime import datetime

        start_time = datetime.now().isoformat()

        # Save config at start
        self.save_config_metadata()

        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.config.output_dir}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            self.train_epoch()

            # Validation
            if (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.validate()

                # Save best model
                if val_metrics["total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total"]
                    self.save_checkpoint("best.pt")
                    print(f"  New best model! Val loss: {val_metrics['total']:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

            # Step scheduler
            if self.scheduler and self.config.scheduler != "onecycle":
                self.scheduler.step()

        print("Training complete!")

        # Save metadata at end
        self.save_run_metadata(start_time)

        self.writer.close()

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict with average training metrics
        """
        self.model.train()
        epoch_losses: dict[str, list[float]] = {
            "total": [],
            "boundary": [],
            "energy": [],
            "beat": [],
            "label": [],
        }

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            # Forward pass with AMP
            with autocast("cuda", enabled=self.use_amp):
                predictions = self.model(batch["audio"])

            # Convert predictions to fp32 for loss computation
            if self.use_amp:
                predictions = {k: v.float() for k, v in predictions.items()}

            # Compute loss (outside autocast due to BCE restrictions)
            targets = {
                "boundary": batch["boundary"],
                "energy": batch["energy"],
                "beat": batch["beat"],
                "label": batch["label"],
            }
            losses = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                # Scaled backward pass for AMP
                self.scaler.scale(losses["total"]).backward()

                # Gradient clipping with unscaling
                if self.config.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard backward pass
                losses["total"].backward()

                # Gradient clipping
                if self.config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.optimizer.step()

            # Step scheduler (for onecycle)
            if self.scheduler and self.config.scheduler == "onecycle":
                self.scheduler.step()

            # Log losses
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value.item())

            # Periodic logging
            if (batch_idx + 1) % self.config.log_every == 0:
                self._log_batch(batch_idx, losses)

            self.global_step += 1

        # Compute epoch averages
        avg_losses = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in epoch_losses.items()
        }

        # Log epoch metrics
        self._log_epoch(avg_losses, prefix="train")

        return avg_losses

    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dict with average validation metrics
        """
        self.model.eval()
        val_losses: dict[str, list[float]] = {
            "total": [],
            "boundary": [],
            "energy": [],
            "beat": [],
            "label": [],
        }

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                predictions = self.model(batch["audio"])

                # Compute loss
                targets = {
                    "boundary": batch["boundary"],
                    "energy": batch["energy"],
                    "beat": batch["beat"],
                    "label": batch["label"],
                }
                losses = self.loss_fn(predictions, targets)

                # Accumulate losses
                for key, value in losses.items():
                    if key in val_losses:
                        val_losses[key].append(value.item())

        # Compute averages
        avg_losses = {
            key: sum(values) / len(values) if values else 0.0 for key, values in val_losses.items()
        }

        # Log validation metrics
        self._log_epoch(avg_losses, prefix="val")

        return avg_losses

    def _log_batch(self, batch_idx: int, losses: dict[str, torch.Tensor]) -> None:
        """Log batch metrics.

        Args:
            batch_idx: Batch index
            losses: Loss dict
        """
        loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in losses.items()])
        print(
            f"Epoch {self.current_epoch+1} [{batch_idx+1}/{len(self.train_loader)}] " f"{loss_str}"
        )

    def _log_epoch(self, metrics: dict[str, float], prefix: str = "train") -> None:
        """Log epoch metrics.

        Args:
            metrics: Metrics dict
            prefix: Metric prefix (train/val)
        """
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}_loss", value, self.current_epoch)

        # Log learning rate
        if self.optimizer.param_groups:
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/learning_rate", lr, self.current_epoch)

        # Print summary
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {self.current_epoch+1} {prefix}: {loss_str}")

    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint atomically.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoints_dir / filename
        temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Write to temp file first
        torch.save(checkpoint, temp_path)

        # Atomic rename to final location
        import os

        os.replace(temp_path, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_config_metadata(self) -> None:
        """Save training configuration to config.yaml."""
        import yaml

        config_path = self.config.output_dir / "config.yaml"
        config_dict = {
            "run_name": self.config.output_dir.name,
            "backbone": getattr(self.model.backbone, "model_name", "unknown"),
            "num_epochs": self.config.num_epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "scheduler": self.config.scheduler,
            "gradient_clip": self.config.gradient_clip,
            "save_every": self.config.save_every,
            "eval_every": self.config.eval_every,
            "device": str(self.device),
        }

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def save_run_metadata(self, start_time: str) -> None:
        """Save run metadata at training completion.

        Args:
            start_time: ISO format start time
        """
        import subprocess
        from datetime import datetime

        import yaml

        end_time = datetime.now().isoformat()

        # Get git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_commit = "unknown"

        # Get data.dvc hash
        data_dvc_path = Path("data.dvc")
        data_hash = None
        if data_dvc_path.exists():
            try:
                with open(data_dvc_path) as f:
                    dvc_data = yaml.safe_load(f)
                    data_hash = dvc_data.get("outs", [{}])[0].get("md5")
            except Exception:
                pass

        metadata = {
            "run_name": self.config.output_dir.name,
            "start_time": start_time,
            "end_time": end_time,
            "final_metrics": {
                "best_epoch": self.current_epoch,
                "best_val_loss": float(self.best_val_loss),
            },
            "dataset": {
                "data_dvc_hash": data_hash,
            },
            "git_commit": git_commit,
            "device": str(self.device),
        }

        metadata_path = self.config.output_dir / "metadata.yaml"
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def create_trainer(
    model: MultiTaskModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    **config_kwargs: Any,
) -> Trainer:
    """Factory function to create a trainer.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        output_dir: Output directory
        **config_kwargs: Additional config arguments

    Returns:
        Configured Trainer
    """
    config = TrainingConfig(output_dir=output_dir, **config_kwargs)
    return Trainer(model, train_loader, val_loader, config)

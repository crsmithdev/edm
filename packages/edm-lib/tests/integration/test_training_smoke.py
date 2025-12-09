"""Smoke test for training pipeline - ensures training can complete without crashing.

This test uses minimal data (2 tracks, 1 batch, 1 epoch) and is designed to catch:
- Model architecture bugs (shape mismatches, missing layers)
- Dataset/dataloader issues (collation, batching)
- Loss computation errors
- Checkpoint saving/loading
- Label vocabulary auto-detection

Runtime: ~10-30 seconds on CPU, ~5 seconds on GPU
Marked as @slow - run with: pytest -m slow or pytest tests/integration/
"""

from pathlib import Path

import pytest
import torch
import yaml
from edm.models.multitask import create_model
from edm.training import create_dataloaders
from edm.training.losses import MultiTaskLoss
from edm.training.trainer import Trainer, TrainingConfig


@pytest.fixture
def minimal_annotations(tmp_path: Path) -> Path:
    """Create minimal annotation dataset (2 tracks) for smoke testing."""
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    # Annotation 1: Simple structure with standard labels
    annotation1 = {
        "metadata": {
            "tier": 2,
            "confidence": 0.9,
            "source": "manual",
            "annotator": "smoke_test",
        },
        "audio": {
            "file": "test1.mp3",
            "duration": 30.0,
            "bpm": 128.0,
            "downbeat": 0.0,
            "time_signature": [4, 4],
        },
        "structure": [
            {"bar": 1, "label": "intro", "time": 0.0, "confidence": 1.0},
            {"bar": 9, "label": "buildup", "time": 15.0, "confidence": 1.0},
            {"bar": 17, "label": "drop", "time": 30.0, "confidence": 1.0},
        ],
    }

    # Annotation 2: Include 'unlabeled' to test vocabulary building
    annotation2 = {
        "metadata": {
            "tier": 2,
            "confidence": 0.8,
            "source": "manual",
            "annotator": "smoke_test",
        },
        "audio": {
            "file": "test2.mp3",
            "duration": 30.0,
            "bpm": 140.0,
            "downbeat": 0.0,
            "time_signature": [4, 4],
        },
        "structure": [
            {"bar": 1, "label": "unlabeled", "time": 0.0, "confidence": 0.5},
            {"bar": 9, "label": "breakdown", "time": 15.0, "confidence": 1.0},
            {"bar": 17, "label": "outro", "time": 30.0, "confidence": 1.0},
        ],
    }

    # Save annotations
    with open(annotations_dir / "test1.yaml", "w") as f:
        yaml.safe_dump(annotation1, f)
    with open(annotations_dir / "test2.yaml", "w") as f:
        yaml.safe_dump(annotation2, f)

    return annotations_dir


@pytest.fixture
def minimal_audio(tmp_path: Path) -> Path:
    """Create minimal audio files (5 seconds, silence) for smoke testing."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Generate 5 seconds of silence at 22050 Hz
    import numpy as np
    import soundfile as sf

    silence = np.zeros(int(5 * 22050), dtype=np.float32)

    sf.write(audio_dir / "test1.mp3", silence, 22050)
    sf.write(audio_dir / "test2.mp3", silence, 22050)

    return audio_dir


@pytest.mark.slow
@pytest.mark.integration
def test_training_smoke(minimal_annotations: Path, minimal_audio: Path, tmp_path: Path) -> None:
    """Smoke test: Train for 1 epoch with minimal data to catch crashes.

    This test verifies:
    - Dataset can load annotations and build label vocabulary
    - Model can be created with auto-detected num_classes
    - Forward pass completes without shape errors
    - Loss computation works for all enabled heads
    - Backward pass and optimizer step succeed
    - Checkpoint can be saved and loaded
    """
    output_dir = tmp_path / "training_output"

    # Create dataloaders (2 tracks, batch_size=2 = 1 batch)
    train_loader, val_loader = create_dataloaders(
        annotation_dir=minimal_annotations,
        audio_dir=minimal_audio,
        batch_size=2,
        train_split=1.0,  # Use all data for training (no validation split needed)
        num_workers=0,  # Single-threaded for determinism
        duration=5.0,  # Match audio length
    )

    # Verify dataset loaded correctly
    dataset = train_loader.dataset.dataset  # type: ignore[attr-defined]
    assert dataset.num_classes == 7, "Should detect 7 unique labels"
    assert len(dataset.label_vocab) == 7
    assert "unlabeled" in dataset.label_vocab

    # Create minimal model (MERT-95M for raw audio support)
    # CNN requires mel spectrograms, MERT accepts raw audio
    model = create_model(
        backbone_type="mert-95m",
        enable_boundary=True,
        enable_energy=False,  # Skip for speed
        enable_beat=False,  # Skip for speed
        enable_label=True,
        num_classes=dataset.num_classes,
    )

    # Use CPU for determinism (GPU if available for speed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Create loss function
    loss_fn = MultiTaskLoss(
        boundary_weight=1.0,
        energy_weight=0.0,
        beat_weight=0.0,  # Beat head disabled
        label_weight=1.0,
    )

    # Create trainer config
    config = TrainingConfig(
        output_dir=output_dir,
        run_name="smoke_test",
        num_epochs=1,
        learning_rate=1e-3,
        log_every=1,  # Log every batch
        eval_every=1,  # Eval every epoch
        save_every=1,  # Save every epoch
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,  # Reuse train as val for smoke test
        config=config,
        loss_fn=loss_fn,
    )

    # Run training (should complete without errors)
    trainer.train()

    # Verify outputs
    checkpoint_dir = output_dir / "checkpoints"
    assert checkpoint_dir.exists(), "Checkpoint directory should be created"
    assert (checkpoint_dir / "best.pt").exists(), "Best checkpoint should be saved"

    # Verify checkpoint can be loaded
    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "epoch" in checkpoint
    # Epoch might be 0-indexed or 1-indexed, just verify it's a valid number
    assert checkpoint["epoch"] >= 0

    # Verify model can be loaded from checkpoint
    loaded_model = create_model(
        backbone_type="mert-95m",
        enable_boundary=True,
        enable_energy=False,
        enable_beat=False,
        enable_label=True,
        num_classes=dataset.num_classes,
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model = loaded_model.to(device)

    # Verify loaded model can run inference
    batch = next(iter(train_loader))
    audio = batch["audio"].to(device)
    with torch.no_grad():
        outputs = loaded_model(audio)

    assert "boundary" in outputs
    assert "label" in outputs
    assert outputs["label"].shape[-1] == dataset.num_classes, "Output should match num_classes"


@pytest.mark.slow
@pytest.mark.integration
def test_label_vocabulary_flexibility(minimal_annotations: Path, minimal_audio: Path) -> None:
    """Test that label vocabulary adapts to dataset labels.

    Verifies the fix for hardcoded num_classes - system should auto-detect
    the number of unique labels and configure the model accordingly.
    """
    # Create dataset
    from edm.training.dataset import EDMDataset

    dataset = EDMDataset(
        annotation_dir=minimal_annotations,
        audio_dir=minimal_audio,
        duration=5.0,
    )

    # Verify vocabulary was built
    assert hasattr(dataset, "label_vocab"), "Dataset should have label_vocab"
    assert hasattr(dataset, "num_classes"), "Dataset should have num_classes"
    assert dataset.num_classes == len(dataset.label_vocab)

    # Verify all expected labels are present
    expected_labels = {"intro", "buildup", "drop", "breakdown", "outro", "unlabeled"}
    assert set(dataset.label_vocab.keys()) == expected_labels

    # Verify label indices are consistent
    for label, idx in dataset.label_vocab.items():
        assert 0 <= idx < dataset.num_classes

    # Verify no duplicate indices
    indices = list(dataset.label_vocab.values())
    assert len(indices) == len(set(indices)), "Label indices should be unique"

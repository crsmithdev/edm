# [MLPIVOT] Implementation Tasks

## Phase 1: Data Pipeline

- [ ] 1.1 Create DJ label parser module
  - [ ] 1.1.1 Rekordbox XML parser
  - [ ] 1.1.2 Common interface for label sources
  - [ ] 1.1.3 Cue point to timestamp conversion

- [ ] 1.2 Implement data validation
  - [ ] 1.2.1 Beat grid snapping check
  - [ ] 1.2.2 Duration sanity check
  - [ ] 1.2.3 Format validation

- [ ] 1.3 Create dataset splits
  - [ ] 1.3.1 Define annotation schema for training labels
  - [ ] 1.3.2 Script to generate train/val/test splits
  - [ ] 1.3.3 Tier assignment logic (clean vs noisy)

## Phase 2: Feature Extraction

- [ ] 2.1 Implement backbone loading
  - [ ] 2.1.1 MERT model wrapper
  - [ ] 2.1.2 Configurable layer freezing
  - [ ] 2.1.3 GPU/CPU device handling

- [ ] 2.2 Create feature pipeline
  - [ ] 2.2.1 Audio loading and chunking
  - [ ] 2.2.2 Mel spectrogram computation
  - [ ] 2.2.3 Feature caching for training

## Phase 3: Model Implementation

- [ ] 3.1 Build prediction heads
  - [ ] 3.1.1 Boundary head (frame-wise classification)
  - [ ] 3.1.2 Energy head (multi-band regression)
  - [ ] 3.1.3 Optional event head (drop detection)

- [ ] 3.2 Implement multi-task model
  - [ ] 3.2.1 Combined model class
  - [ ] 3.2.2 Forward pass with all heads
  - [ ] 3.2.3 Model serialization/loading

## Phase 4: Training Loop

- [ ] 4.1 Create dataset class
  - [ ] 4.1.1 DJ label dataset implementation
  - [ ] 4.1.2 Data augmentation (pitch shift, time stretch)
  - [ ] 4.1.3 Batching with variable-length sequences

- [ ] 4.2 Implement losses
  - [ ] 4.2.1 Boundary-tolerant BCE
  - [ ] 4.2.2 Energy MSE
  - [ ] 4.2.3 Multi-task weighting

- [ ] 4.3 Training script
  - [ ] 4.3.1 Training loop with logging
  - [ ] 4.3.2 Validation metrics
  - [ ] 4.3.3 Checkpointing

## Phase 5: Label Quality

- [ ] 5.1 Integrate cleanlab
  - [ ] 5.1.1 Cross-validation prediction script
  - [ ] 5.1.2 Error detection workflow
  - [ ] 5.1.3 Report generation for review

- [ ] 5.2 Noise-robust training
  - [ ] 5.2.1 Loss reweighting implementation
  - [ ] 5.2.2 Label smoothing
  - [ ] 5.2.3 Co-teaching (optional)

## Phase 6: Inference Integration

- [ ] 6.1 Create ML detector class
  - [ ] 6.1.1 Implement StructureDetector protocol
  - [ ] 6.1.2 Model loading with fallback
  - [ ] 6.1.3 Chunked inference for long files

- [ ] 6.2 Post-processing
  - [ ] 6.2.1 Peak picking for boundaries
  - [ ] 6.2.2 Beat grid snapping
  - [ ] 6.2.3 Minimum section filtering

- [ ] 6.3 Wire into analysis pipeline
  - [ ] 6.3.1 Add to get_detector() factory
  - [ ] 6.3.2 CLI flag for detector selection
  - [ ] 6.3.3 Default detector configuration

## Phase 7: Evaluation

- [ ] 7.1 Extend evaluation framework
  - [ ] 7.1.1 Boundary F1 at various tolerances
  - [ ] 7.1.2 Energy correlation metrics
  - [ ] 7.1.3 Per-genre breakdown

- [ ] 7.2 Benchmarking
  - [ ] 7.2.1 Compare ML vs MSAF vs energy detector
  - [ ] 7.2.2 Ablation studies (backbone, heads, data size)
  - [ ] 7.2.3 Document results

## Dependencies

**Required packages:**
```
transformers>=4.30.0  # MERT
cleanlab>=2.0.0
torchaudio>=2.0.0
```

**Optional (better performance):**
```
flash-attn  # Faster transformer attention
```

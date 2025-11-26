# Project Context

## Purpose
An ML-powered system for comprehensive musical and structural analysis of EDM tracks. The system focuses on:
- Essential structural elements (BPM detection and beat grid alignment)
- Key track moments (drops, builds, breakdowns, risers)
- DJ-specific features (mix points, loop regions, phrase boundaries)
- Energy and tension profiling for harmonic analysis
- External connectivity for DJ software integration

**Current Status:** Early-stage planning phase. Architecture designed, implementation pending.

## Tech Stack

### Core Audio Processing
- **librosa** - Spectral features, onset detection, tempo tracking (fallback BPM detector)
- **beat_this** - Primary BPM detector (ISMIR 2024, neural network beat tracker from CPJKU)
- **essentia** - EDM-specific algorithms for key detection, danceability, energy analysis

### Machine Learning
- **PyTorch** or **TensorFlow** - Deep learning framework
- **torchaudio** - GPU-accelerated spectrograms (if using PyTorch)

### Development Environment
- **Python 3.10+** - Primary language
- **pytest** - Testing framework
- **black** - Code formatter
- **ruff** - Linter
- **Jupyter notebooks** - Experimentation and visualization

### Planned Directory Structure
```
edm/
├── src/
│   ├── models/          # Neural network architectures
│   ├── features/        # Audio feature extraction
│   ├── data/            # Data loading and augmentation
│   ├── training/        # Training loops and utilities
│   └── analysis/        # Track analysis and inference
├── tests/               # Unit and integration tests
├── data/                # Training data and datasets
├── notebooks/           # Jupyter notebooks for experimentation
├── configs/             # Model and training configurations
└── docs/                # Documentation
```

## Project Conventions

### Code Style
- **Modular Design**: Keep files under 500 lines
- **Type Hints**: Use Python type annotations throughout
- **Docstrings**: NumPy-style docstrings for all public functions
- **Formatting**: black (default settings)
- **Linting**: ruff with EDM-specific rules
- **Imports**: Organized by stdlib, third-party, local
- **No Hardcoded Values**: Use configuration files for all parameters

### Architecture Patterns

**Multi-Model Architecture** with specialized components:
1. **Structural Segmentation Model (CNN + RNN)** - Detects intro/breakdown/drop/build/outro
2. **Drop Detection Specialist (TCN)** - Focuses on bass drops using multi-resolution spectrograms
3. **BPM/Beat Grid Model** - Combines traditional DSP with learned refinements
4. **Energy/Tension Profiler** - Analyzes energy bands and tension curves

**Design Principles:**
- **CNN** captures local rhythmic patterns (kick-snare, hi-hat rhythms)
- **RNN/Mamba** understands metrical hierarchy and long-range dependencies
- **Multi-scale processing** handles both micro-timing and phrase structure
- **Differentiable DSP layers (DDSP)** combine learned and traditional signal processing
- Consider **Mamba/State Space Models** for linear-time complexity on full tracks

**Clean Architecture:**
- Separate feature extraction from model inference
- Dependency injection for model selection
- Strategy pattern for different beat detection algorithms
- Factory pattern for model instantiation

### Testing Strategy

**Test-First Development:**
- Write tests before implementation
- Target 90%+ code coverage
- Separate test types into distinct directories

**Testing Layers:**
1. **Unit Tests** - Individual feature extraction functions, model components
2. **Integration Tests** - Full analysis pipeline on sample tracks
3. **Regression Tests** - Maintain accuracy on benchmark tracks as code evolves
4. **Perceptual Tests** - Validate predictions against hand-tagged ground truth (±2 second tolerance)

**Validation Approach:**
- Precision/recall for boundaries with ±2 second tolerance windows
- Perceptual validation through A/B testing in DJ sets
- Cross-validation by subgenre (progressive house, melodic techno, psytrance)

### Git Workflow
- **Main Branch**: Production-ready code only
- **Feature Branches**: `feature/description` for new features
- **Test Branches**: `test/description` for testing improvements
- **Commit Style**: Conventional commits (feat:, fix:, test:, docs:, refactor:)
- **PR Requirements**: All tests pass, code reviewed, no decrease in coverage

## Domain Context

### EDM-Specific Knowledge

**Beat Detection Philosophy:**
- Use beat_this neural network as primary (ISMIR 2024, state-of-the-art accuracy)
- Detect and lock to kick drums - kick is ground truth in EDM
- Validate beat grids against structural boundaries (drops align with downbeats)
- Check tempo stability - real EDM doesn't drift significantly

**Analysis Targets:**
- **Structural**: BPM, beat grids, phrasing, key, track sections
- **Key Moments**: Drops, builds, breakdowns, risers/sweeps, bass cuts/returns
- **Transitions**: Percussion rolls/fills, fake drops, switch-ups
- **DJ Features**: Mix-in/mix-out zones, loop-safe regions, mixing hazards, phrase boundaries
- **Energy/Harmonic**: Energy plateaus vs peaks, tension curves, key modulations

**Feature Engineering Priorities:**
- **Time-domain**: Zero-crossing rate, RMS energy, peak/average ratios
- **Frequency-domain**: Spectral centroid, spectral rolloff, harmonic-percussive separation
- **Music-specific**: Onset strength patterns, chroma energy, tempogram

### Recent ML Techniques to Consider
- Self-supervised contrastive learning (CLMR, COLA)
- Neural audio codecs (Encodec, DAC) as feature representations
- Mamba/State Space Models for efficient long-sequence modeling
- Mixture-of-Experts for multi-genre handling
- Hierarchical Transformers (Perceiver AR) for multi-timescale processing
- Few-shot learning (Prototypical Networks) for rare patterns
- Active learning with uncertainty (Monte Carlo dropout)

## Important Constraints

**Performance:**
- Models must run on consumer GPUs (RTX 3060+) or CPU-only environments
- Track analysis should complete in under 30 seconds per song
- Memory footprint under 4GB during inference

**Environment Safety:**
- Never hardcode API keys or credentials
- Use environment variables for sensitive configuration
- Validate all external audio file inputs

**Data Privacy:**
- No tracking of user music libraries
- All analysis performed locally by default
- Optional cloud features must be opt-in

**Accuracy Requirements:**
- BPM detection: ±0.5 BPM accuracy for stable tempo tracks
- Drop detection: Precision >90%, Recall >85%
- Beat grid alignment: ±50ms tolerance at drop points

## External Dependencies

**Pre-trained Models:**
- Pre-training on FMA (Free Music Archive), MagnaTagATune
- Transfer learning to EDM-specific datasets
- Active learning loop for continuous improvement

**Data Augmentation:**
- Pitch shifting (±2 semitones) without tempo change
- Time stretching (95-105% tempo variation)
- Mix simulation - overlay tracks at different volumes
- EQ augmentation - DJ-style EQ curves
- Pseudo-labeling for high-confidence predictions

**Future Integrations:**
- Rekordbox XML export
- Traktor NML format
- Serato markers
- Engine DJ database
- Ableton Live integration

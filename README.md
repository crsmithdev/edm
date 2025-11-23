# EDM (temporary handle for a new, untitled project)

 claude-flow swarm "read docs\PROJECT_SPEC.md and create a detailed technical implementation plan in /plans using TDD, the SAFLA model, in Python, using a VSCode DevContainer and WSL, using pip and uv.  Focus first on accurate BPMs, beat grids and phrasing.  Set up a mount for music files in the DevContainer, mapped to C:\Music\Library on the host machine.  Ensure claude-code and other AI tooling work properly in the DevContainer.  Just do research, do not write code yet or implemnent anything.

## Vision

An ML-powered system for comprehensive musical and strucural analysis of EDM tracks.  This is part of a larger project.  Initially, the focus is entirely on essential structural elements (BPM), key track moments (drops, builds, breakdowns) and external connectivity.

## Planned Features

**Structural Elements:**
- **Essentials** - BPM, beat grids, phrasing, key
- **Track progression** - intro, verse, chorus, bridge, outro, etc.
- **Key moments** - drops, builds and breakdowns
- **Risers/Sweeps** - The specific automation c     ves (exponential vs linear) and their duration
- **Percussion rolls/fills** - Snare rolls, hi-hat patterns that signal transitions (often at bars 7-8 or 15-16)
- **Bass cuts/returns** - The specific pattern of bass removal (full cut vs high-pass sweep)
- **Vocal chops/hooks** - Repetitive vocal elements that DJs can use as mixing anchors
- **"Fake drops"** - Where tension builds but releases into another breakdown instead
- **Switch-ups** - Mid-phrase rhythm changes common in psytrance and tech-house

**Harmonic/Melodic Structure:**
- **Key modulation points** - Tracks that shift key (important for harmonic mixing)
- **Melodic call-and-response** patterns 
- **Chord progression loops** - 4-bar vs 8-bar vs 16-bar harmonic cycle
- **Tonal vs atonal sections** - Some tracks strip to pure percussion
- **Harmonic density** - Stacked leads vs single elements (affects EQ headroom for mixing)

**Energy Topology:**
- **Energy "plateaus" vs "peaks"** - Some drops maintain energy, others spike and decay
- **Tension release patterns** - Immediate vs gradual, full vs partial
- **"Breathing" patterns** - Regular energy oscillations within sections
- **Drive patterns** - Whether energy increases linearly or in steps

**DJ-Specific Markers:**
- **Mix-in/mix-out zones** - Sections specifically designed for blending
- **Loop-safe regions** - Areas without progression that can loop indefinitely
- **Mixing hazards** - vocal sections, unusual structural changes, etc.
- **Phrase vulnerability** - Sections where losing phase alignment is catastrophic vs forgiving
- **EQ pockets** - Frequency ranges intentionally left empty for layering
- **"DJ tool" sections** - Minimal loops meant for creative mixing

**External Data**
- **Local** -  extract BPM, key and other relevant metadata from files
- **Online** - fetch data from services  / APIs (Spotify, etc.) to find / validate track data
- **Priority** - for basic things (BPM, key) check local, then online, then analyze if still missing

## Core Software Stack

**Audio Processing & Feature Extraction:**
- **librosa** (which you're already using) for spectral features, onset detection, tempo tracking
- **madmom** - specifically designed for music information retrieval, excellent for beat/downbeat tracking
- **essentia** - has EDM-specific algorithms for key detection, danceability, and energy analysis
- **pytorch** or **tensorflow** for the deep learning components
- **torchaudio** if using PyTorch - provides GPU-accelerated spectrograms

## Multi-Model Architecture

This will require several specialized models working together:

### 1. **Structural Segmentation Model (CNN + RNN)**
- Input: Mel-spectrograms, chromagrams, self-similarity matrices
- Architecture: CNN for local pattern detection → Bi-LSTM for temporal context → CRF layer for segment boundaries
- Output: Frame-level predictions of intro/breakdown/drop/build/outro
- Training: Your hand-tagged timestamps of structural boundaries

### 2. **Drop Detection Specialist**
- Input: Multi-resolution spectrograms (focusing on 20-200Hz for bass drops)
- Features: RMS energy derivatives, spectral flux, sub-bass energy ratios
- Architecture: Temporal Convolutional Network (TCN) - better than RNNs for long sequences
- Key insight: Train on 16-32 bar windows centered on drops vs non-drops

### 3. **BPM/Beat Grid Model**
- Start with traditional DSP (autocorrelation, comb filters) for initial estimate
- Refine with learned model that handles tempo changes and half/double-time sections
- Train on tracks where you've manually corrected Rekordbox's beat grids

### 4. **Energy/Tension Profiler**
- Extract energy bands: sub-bass (20-60Hz), bass (60-250Hz), mid (250-4kHz), high (4kHz+)
- Model tension curves using attention mechanisms to identify build patterns
- Output: Continuous energy/tension values that inform your cue point placement

## Feature Engineering

**Time-domain features:**
- Zero-crossing rate (percussion density)
- RMS energy with multiple window sizes
- Peak/average ratios (dynamic range)

**Frequency-domain features:**
- Spectral centroid movement (brightness changes)
- Spectral rolloff points
- Harmonic-percussive separation ratios

**Music-specific features:**
- Onset strength patterns (for identifying percussion strips/fills)
- Chroma energy normalized statistics (harmonic stability)
- Tempogram (for detecting rhythm changes)

## Training Data & Data Augmentation Strategies

I will provide a set of hand-tagged tracks for training as needed.

Since hand-tagging is expensivem, these methods may be useful:

- **Pitch shifting** (±2 semitones) without tempo change
- **Time stretching** (95-105% tempo) 
- **Mix simulation** - overlay tracks at different volumes to simulate mixing scenarios
- **EQ augmentation** - apply DJ-style EQ curves
- **Pseudo-labeling** - use high-confidence predictions to expand training set

## Training Approach

1. **Start with pre-training** on large music datasets (FMA, MagnaTagATune) for general music understanding
2. **Transfer learning** to EDM-specific features using your hand-tagged data
3. **Active learning loop**:
   - Model makes predictions on new tracks
   - You correct only the uncertain/wrong predictions
   - Retrain with expanded dataset
   - Your existing 244-track library could seed this effectively

4. **Multi-task learning** - train all objectives jointly with shared encoder:
   - Structure segmentation (classification)
   - BPM prediction (regression)
   - Cue point suggestion (sequence labeling)
   - Energy profiling (regression)

## Validation Strategy

- **Objective metrics**: Precision/recall for boundaries (±2 seconds tolerance)
- **Perceptual validation**: A/B test cue points in actual DJ sets
- **Cross-validation by subgenre**: Ensure it works across progressive house, melodic techno, psytrance

The CNN+RNN architecture works because:
- **CNN** captures local rhythmic patterns (kick-snare patterns, hi-hat rhythms)
- **RNN** understands the metrical hierarchy (which beats are strong/weak)
- **Multi-scale** processing handles both micro-timing and phrase structure

## Key Insights for Maximum Reliability

1. **Always use Madmom's DBN tracker as your primary** - It's specifically trained on electronic music
2. **Detect and lock to kick drums** - In EDM, the kick is the truth
3. **Validate with structure** - Drops should align with downbeats
4. **Learn from your corrections** - Build a personalized correction model
5. **Check tempo stability** - Real EDM doesn't drift; if it does, you've got the wrong grid

## Relevant Recent ML Techniques

### 1. **Diffusion Models for Audio (2023-2024)**
- Not just for generation - can be used for **masked audio modeling**
- Train to reconstruct missing segments → learns deep structure
- Recent work: Moûsai, AudioLDM2 architectures adapted for analysis

### 2. **Hierarchical Transformers**
- **HiP-Hop** (Hierarchical Prosody Prediction) techniques from speech can map to music
- **Perceiver AR** - Handles multiple timescales naturally
- **FlashAttention-2** - Makes it feasible to process full tracks at sample level

### 3. **Self-Supervised Contrastive Learning**
- **CLMR** (Contrastive Learning for Music Representations)
- **SimCLR adapted for audio** - Learn representations without labels
- **COLA** (Contrastive Learning for Audio) - Specifically good for structure
- Train on augmented pairs: same track with different EQ = similar, different tracks = dissimilar

### 4. **Neural Audio Codecs as Features**
- **Encodec**, **SoundStream** representations capture perceptual importance
- Much better than raw spectrograms for downstream tasks
- **DAC (Descript Audio Codec)** - Optimized for music, not just speech

### 5. **Mamba/State Space Models (2024)**
- Linear-time complexity for long sequences (perfect for full tracks)
- **Audio-Mamba** architectures showing SOTA results
- Better than transformers for catching long-range dependencies in music

### 6. **Multimodal Music Understanding**
- **MusicFM** - Foundation model trained on music + metadata + user behavior
- **CLAP** models (like CLIP but for audio/text)
- Could incorporate DJ comments, crowd recordings, social media reactions

### 7. **Differentiable DSP Layers**
- **DDSP** (Differentiable Digital Signal Processing)
- Combine learned and traditional signal processing
- Example: Learned filterbanks that discover EDM-relevant frequency bands

### 8. **Mixture-of-Experts for Multi-Genre**
- Different experts for different subgenres
- Router network learns track style and activates appropriate experts
- Particularly good for handling progressive → techno → psytrance differences

## Specific Recent Papers/Techniques Worth Implementing

**"Beat Transformer" (2023)** - Attention-based beat tracking that handles complex patterns:
```python
# Learns to attend to different metrical levels
class BeatTransformer(nn.Module):
    def forward(self, audio_features):
        # Multi-head attention across different time scales
        beat_attention = self.beat_heads(audio_features)
        downbeat_attention = self.downbeat_heads(audio_features)
        phrase_attention = self.phrase_heads(audio_features)
        return hierarchical_decode(beat_attention, downbeat_attention, phrase_attention)
```

**"MusCALL" (2024)** - Curriculum learning for structure:
- Start training on simple 4/4 house
- Gradually introduce complex prog house and psytrance
- Model learns robust features that generalize

**"ProtoTypical Networks for Audio"** - Few-shot learning:
- Define prototypes for "drop", "breakdown", etc. from just a few examples
- Useful when you have limited tagged data for rare patterns

### Integration Ideas

**Ensemble with Music Theory Priors:**
```python
class EDMStructureNet(nn.Module):
    def __init__(self):
        # Neural components
        self.mamba_encoder = MambaBlock(d_model=512)
        self.perceiver = PerceiverResampler()
        
        # Theory-informed components
        self.bar_attention = BarAlignedAttention()  # Forces attention at 8/16 bar boundaries
        self.energy_physics = EnergyConservationLoss()  # Energy can't jump discontinuously
```

**Active Learning with Uncertainty:**
- Use Monte Carlo dropout to identify tracks where model is uncertain
- Focus your hand-tagging efforts on these high-value examples
- Recent work shows 10x reduction in labeling needs

The biggest game-changer could be combining Mamba/SSM architectures (for efficient long-range modeling) with differentiable DSP (for interpretable features) and contrastive pre-training on large unlabeled EDM collections. This would give a foundation model that understands EDM deeply before even adding labels.
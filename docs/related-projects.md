# DJ/Music Analysis Software on GitHub

Based on your EDM Track Analysis project (BPM detection, structure analysis using neural networks), here's a comprehensive list of similar projects organized by category.

---

## Full DJ Software

### [Mixxx](https://github.com/mixxxdj/mixxx) ⭐ **Highly Relevant**
The only fully open-source DJ software with integrated analysis. Uses libkeyfinder for key detection and VAMP plugins (including aubio) for beat/tempo detection.
- **Approach**: Plugin-based architecture allowing swappable algorithms
- **Accuracy**: No formal benchmark published, but uses battle-tested libraries
- **Learn from**: Their plugin architecture for algorithm swapping, BPM/key override UI

---

## Beat/Tempo Detection

### [beat_this](https://github.com/CPJKU/beat_this) ⭐⭐ **Most Relevant**
State-of-the-art beat/downbeat tracker from ISMIR 2024. This is the library your project uses.
- **Approach**: CNN+Transformer architecture, NO Dynamic Bayesian Network postprocessing
- **Accuracy**: Surpasses SOTA on F1 score across 16 benchmark datasets (ASAP, Ballroom, Beatles, Harmonix, etc.)
- **Evaluation**: Uses mir_eval metrics, trained on multiple datasets for generalization
- **Learn from**: Their tolerance-aware loss function for annotation timing errors; multi-dataset training strategy

### [BeatNet](https://github.com/mjhydri/BeatNet) ⭐⭐ **Highly Relevant**
Real-time and offline beat/downbeat/tempo/meter tracking using CRNN + particle filtering.
- **Approach**: CRNN for activation function, particle filtering for beat inference
- **Accuracy**: ISMIR 2021 paper implementation, claims real-time SOTA
- **Modes**: Streaming (microphone), real-time (file), offline
- **Learn from**: Real-time capability via particle filtering; combines librosa (features) + madmom (state space)

### [madmom](https://github.com/CPJKU/madmom) ⭐⭐ **Highly Relevant**
Python audio/MIR library with SOTA beat tracking algorithms.
- **Approach**: RNN-based beat/tempo detection with multiple algorithms
- **Accuracy**: "Accurate Tempo Estimation based on Recurrent Neural Networks and Resonating Comb Filters" (ISMIR 2015) - near-perfect MIREX results
- **Evaluation**: MIREX benchmark compatibility
- **Learn from**: Multiple algorithm implementations; resonating comb filter approach for tempo

### [aubio](https://github.com/aubio/aubio)
C library for onset, beat, tempo detection with Python bindings.
- **Approach**: Traditional DSP + some ML methods
- **Accuracy**: Well-established baseline, used by Mixxx
- **Learn from**: Robustness, cross-platform support, VAMP plugin integration

### [Essentia](https://github.com/MTG/essentia) ⭐ **Highly Relevant**
Comprehensive C++ library from Music Technology Group (UPF) with Python bindings.
- **Approach**: 100+ algorithms including RhythmExtractor2013 for beat/tempo
- **Accuracy**: Outputs confidence scores for beat detection
- **Evaluation**: Extensively documented, used in academic research
- **Learn from**: Algorithm diversity; confidence estimation; deep learning model integration

### [librosa](https://github.com/librosa/librosa)
Python library for audio/music analysis.
- **Approach**: Dynamic programming beat tracker
- **Accuracy**: Baseline quality, not SOTA
- **Learn from**: API design patterns; widely used reference implementation

### [tempnetic](https://github.com/csteinmetz1/tempnetic)
Neural network tempo estimation.
- **Approach**: Compares against librosa baseline
- **Evaluation**: Uses MIREX-style accuracy metric (within 8% tolerance)

### [determine_tempo](https://github.com/pnlong/determine_tempo)
Neural network trained on 2000+ songs.
- **Approach**: End-to-end deep learning on MP3 files

---

## Key Detection

### [libKeyFinder](https://github.com/mixxxdj/libkeyfinder) ⭐ **Relevant**
Musical key detection library now maintained by Mixxx.
- **Approach**: Chroma-based key profiling
- **Learn from**: Battle-tested in production DJ software

### [KeyFinder](https://github.com/ibsh/is_KeyFinder)
Desktop application using libKeyFinder.
- **Approach**: MSc research project from 2011

### [keyfinder-cli](https://github.com/evanpurkhiser/keyfinder-cli)
CLI wrapper for libKeyFinder for batch processing.

### [musical-key-finder](https://github.com/jackmcarthur/musical-key-finder)
Python implementation using standard libraries.

---

## Music Structure Analysis

### [all-in-one](https://github.com/mir-aidj/all-in-one) ⭐⭐ **Most Relevant for Structure**
All-In-One Music Structure Analyzer with beat, downbeat, and segment detection.
- **Approach**: Ensemble of 8 models trained on Harmonix Set with 8-fold cross-validation
- **Outputs**: BPM, beats, downbeats, segments (intro/verse/chorus/etc.), embeddings
- **Accuracy**: Uses source separation (Demucs) to create 4-stem embeddings
- **Learn from**: **Ensemble averaging for accuracy**; segment labeling; Harmonix training data

### [msaf](https://github.com/urinieto/msaf)
Music Structure Analysis Framework.
- **Approach**: Multiple algorithms for boundary detection and labeling
- **Learn from**: Framework design for comparing algorithms

### [sf_segmenter](https://github.com/wayne391/sf_segmenter)
Based on Serrà et al. (2012) "Unsupervised Detection of Music Boundaries".
- **Approach**: Structural features, time series analysis

### [SegmentationCNN](https://github.com/mleimeister/SegmentationCNN)
CNN-based music segmentation.
- **Accuracy**: 59% F-measure on SALAMI dataset at 2-beat tolerance
- **Learn from**: Beat-aligned boundary detection

### [MusicBoundariesCNN](https://github.com/carlosholivan/MusicBoundariesCNN)
PyTorch CNN for boundary detection.
- **Evaluation**: SALAMI 2.0 dataset

---

## Source Separation (for Analysis Preprocessing)

### [Demucs](https://github.com/facebookresearch/demucs) ⭐
Facebook's SOTA source separation (drums/bass/vocals/other).
- **Approach**: Hybrid Transformer (v4), won Sony MDX Challenge
- **Learn from**: all-in-one uses Demucs for stem-specific analysis

### [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch)
Reference implementation for source separation.
- **Approach**: Bidirectional LSTM on spectrograms
- **Learn from**: Clean architecture, PyTorch 2.0 compatible

### [Spleeter](https://github.com/deezer/spleeter)
Deezer's original source separation tool.
- **Learn from**: Simple API, widely adopted

---

## Evaluation & Benchmarking

### [mir_eval](https://github.com/mir-evaluation/mir_eval) ⭐⭐ **Critical**
Standard MIR evaluation metrics.
- **Metrics**: Beat F1, tempo accuracy, onset detection, segment boundaries
- **Learn from**: **Implement their metrics for your evaluation framework**

### [GiantSteps Tempo Dataset](https://github.com/GiantSteps/giantsteps-tempo-dataset) ⭐ **EDM-Specific**
EDM-focused tempo/key dataset from ISMIR 2015.
- **Annotations**: User-corrected BPM/key for EDM tracks
- **Learn from**: EDM-specific ground truth; annotation methodology

### [Harmonix Set](https://github.com/urinieto/harmonixset) ⭐
912 Western Pop tracks with beats, downbeats, and structural annotations.
- **Learn from**: Structure annotation schema; training data source

### [JAMS](https://github.com/marl/jams)
JSON annotation format for MIR research.
- **Learn from**: Standardized annotation format

---

## Audio Fingerprinting & Identification

### [Dejavu](https://github.com/worldveil/dejavu)
Shazam-like audio fingerprinting in Python.
- **Accuracy**: 60% with 1 second, 96% with 2 seconds, 100% with 5+ seconds
- **Approach**: Spectrogram peak constellation matching
- **Learn from**: Could identify duplicate tracks or verify metadata

### [Chromaprint](https://github.com/acoustid/chromaprint)
Audio fingerprinting for AcoustID/MusicBrainz.
- **Learn from**: Integration with metadata services

### [Panako](https://github.com/JorenSix/Panako)
Handles pitch shifting/time stretching in fingerprinting.
- **Learn from**: Robust to tempo/pitch modifications

---

## Spotify/Industry Tools

### [Pedalboard](https://github.com/spotify/pedalboard)
Spotify's audio processing library.
- **Learn from**: Data augmentation for training

### [Basic-Pitch](https://github.com/spotify/basic-pitch)
Audio-to-MIDI with pitch bend detection.
- **Approach**: Lightweight neural network
- **Learn from**: Polyphonic pitch detection

---

## Feature Extraction Frameworks

### [Vamp Plugins](https://github.com/vamp-plugins/vamp-plugin-sdk)
Plugin SDK for audio analysis.
- **Learn from**: Mixxx uses this; enables algorithm swapping

### [BBC Vamp Plugins](https://github.com/bbc/bbc-vamp-plugins)
BBC's audio feature extraction collection.

### [QM Vamp Plugins](https://github.com/c4dm/qm-vamp-plugins)
Queen Mary University's collection.

### [Yaafe](https://github.com/Yaafe/Yaafe)
Audio feature extraction toolbox.

---

## Data Conversion Tools

### [dj-data-converter](https://github.com/digital-dj-tools/dj-data-converter)
Convert between Traktor/Rekordbox/Serato formats.
- **Learn from**: Interoperability with commercial DJ software

---

## Highlighted Projects to Learn From

| Project | Why Learn From It |
|---------|-------------------|
| **beat_this** | Tolerance-aware loss, multi-dataset generalization, NO DBN postprocessing |
| **all-in-one** | **Ensemble model averaging** (8 models), stem-based analysis, structure labels |
| **madmom** | Multiple algorithm implementations, near-perfect MIREX accuracy |
| **mir_eval** | Standard evaluation metrics to implement |
| **GiantSteps** | EDM-specific benchmark dataset |
| **BeatNet** | Real-time capability, particle filtering for inference |
| **Essentia** | Confidence estimation, algorithm diversity |
| **Mixxx** | Production architecture, plugin system |

---

## Key Accuracy/Evaluation Approaches Found

1. **Ensemble/Consensus**: all-in-one averages 8 models; improves generalization
2. **Multi-dataset training**: beat_this trains on 16 datasets simultaneously
3. **Tolerance-aware loss**: beat_this uses loss function tolerant to annotation timing shifts
4. **MIREX benchmarks**: Standard 4% or 8% BPM tolerance; Accuracy1/Accuracy2 (octave errors)
5. **mir_eval metrics**: F1 for beats, continuity metrics, onset precision/recall
6. **Source separation preprocessing**: all-in-one analyzes stems separately then combines
7. **Confidence scores**: Essentia outputs detection confidence
8. **Cross-validation**: all-in-one uses 8-fold CV on Harmonix

---

## Sources

- [beat_this](https://github.com/CPJKU/beat_this)
- [BeatNet](https://github.com/mjhydri/BeatNet)
- [madmom](https://github.com/CPJKU/madmom)
- [Essentia](https://github.com/MTG/essentia)
- [all-in-one](https://github.com/mir-aidj/all-in-one)
- [mir_eval](https://github.com/mir-evaluation/mir_eval)
- [GiantSteps Tempo Dataset](https://github.com/GiantSteps/giantsteps-tempo-dataset)
- [Mixxx](https://github.com/mixxxdj/mixxx)
- [libKeyFinder](https://github.com/mixxxdj/libkeyfinder)
- [Demucs](https://github.com/facebookresearch/demucs)
- [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch)
- [msaf](https://github.com/urinieto/msaf)
- [Harmonix Set](https://github.com/urinieto/harmonixset)
- [Dejavu](https://github.com/worldveil/dejavu)
- [Chromaprint](https://github.com/acoustid/chromaprint)
- [JAMS](https://github.com/marl/jams)
- [Pedalboard](https://github.com/spotify/pedalboard)
- [Basic-Pitch](https://github.com/spotify/basic-pitch)
- [aubio](https://github.com/aubio/aubio)

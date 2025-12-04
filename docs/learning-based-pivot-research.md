# Learning-Based Audio Analysis: Research Summary

*Research compiled 2025-12-04*

## Executive Summary

This document captures research on transitioning from algorithmic to learning-based approaches for EDM audio analysis, focusing on practical implementation with limited (hundreds to thousands of tracks) noisy training data from DJ software.

---

## 1. Music Structure Analysis with Deep Learning

### State of the Art: All-In-One Model (ISMIR 2023)

The [All-In-One Music Structure Analyzer](https://github.com/mir-aidj/all-in-one) represents the current SOTA for joint music analysis:

- **Multi-task learning**: Jointly performs beat tracking, downbeat tracking, and functional structure segmentation
- **Architecture**: Dilated neighborhood attentions on source-separated spectrograms
- **Key finding**: "Concurrent learning of beats, downbeats, and segments leads to enhanced performance, with each task mutually benefiting from the others"
- **Labels**: intro, outro, break, bridge, inst, solo, verse, chorus
- **Baseline for MIREX 2025** Music Structure Analysis

**Source**: [arxiv.org/abs/2307.16425](https://arxiv.org/abs/2307.16425)

### CNN-Based Segmentation

Earlier work from Stanford ([cs231n report](https://cs231n.stanford.edu/reports/2016/pdfs/220_Report.pdf)) and the [SegmentationCNN](https://github.com/mleimeister/SegmentationCNN) project established CNN approaches for music segmentation.

A 2024 study in PLOS ONE proposed [supervised metric learning using DNNs](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0312608&type=printable) to extract features for music structure segmentation, distinguishing three basic methods:
- Novelty-based
- Homogeneity-based
- Repetition-based

**Source**: [arxiv.org/abs/2108.12955](https://arxiv.org/abs/2108.12955) - Unsupervised Learning of Deep Features for Music Segmentation

---

## 2. Self-Supervised Learning & Foundation Models

### MERT (Music Understanding Model)

[MERT](https://arxiv.org/pdf/2306.00107) is a self-supervised acoustic music understanding model based on wav2vec 2.0:

- **Training**: Masked language modeling paradigm with acoustic + musical teachers
- **Performance**: Surpasses previous SSL baselines, achieves SOTA on wide range of MIR tasks
- **Advantages**: Smaller parameter size than alternatives
- **Tasks**: Genre classification, emotion prediction, key detection, music tagging

**Source**: [arxiv.org/pdf/2306.00107](https://arxiv.org/pdf/2306.00107)

### MuQ: Mel Residual Vector Quantization (2025)

[MuQ](https://arxiv.org/html/2501.01108v2) is a newer self-supervised music representation model using:
- Mel residual vector quantization
- Improved tokenization over MERT

### Wav2Vec 2.0 for Music

[Research shows](https://arxiv.org/pdf/2210.15310v1) finetuning wav2vec 2.0 pretrained on music achieves competitive results on music classification. However, speech-pretrained wav2vec embeddings have lower performance on music tasks.

### Multi-Level Audio Representations

[IEEE TASLP 2024](https://dl.acm.org/doi/10.1109/TASLP.2024.3379894): "Self-Supervised Learning of Multi-Level Audio Representations for Music Segmentation"
- Contrastive learning maximizes inter-section variance, minimizes intra-section variance
- Training directly optimizes these constraints in latent space

---

## 3. Beat Tracking: Beat This! (ISMIR 2024)

The [Beat This!](https://github.com/CPJKU/beat_this) system represents current SOTA:

### Architecture
- Alternates convolutions with transformers over frequency or time
- **No DBN postprocessing** (unlike traditional approaches)
- Loss function tolerant to small annotation time shifts

### Performance
- Surpasses state of the art in F1 score
- Trained on multiple datasets including solo instruments, time signature changes, classical music

### Training Data Available
- [beat_this_annotations](https://github.com/CPJKU/beat_this_annotations): Beat annotations for 16 datasets
- [Zenodo spectrograms](https://zenodo.org/records/13922116): Preprocessed mel spectrograms published for reproducibility

**Source**: [arxiv.org/abs/2407.21658](https://arxiv.org/abs/2407.21658)

---

## 4. Cue Point Detection with Neural Networks

### CUE-DETR (2024)

[CUE-DETR](https://arxiv.org/html/2407.06823v1) uses object detection for cue point estimation:

- **Model**: Fine-tuned DETR on mel spectrograms
- **Dataset**: EDM-CUE - 4,710 EDM tracks, ~380 hours
  - 95-190 BPM range
  - ~21k manually placed cue points
  - ~4.6 cue points per track
  - **35x larger** than previous cue point datasets
- **Performance**: Higher precision than previous methods
- **Key insight**: High adherence to underlying music structure

**Source**: [arxiv.org/pdf/2407.06823](https://arxiv.org/pdf/2407.06823), [MIT Press](https://direct.mit.edu/comj/article/46/3/67/117159/Automatic-Detection-of-Cue-Points-for-the)

### AI DJ Mix Generator

[GitHub project](https://github.com/sycomix/AI-DJ-Mix-Generator) using:
- LSTM Neural Network (PyTorch)
- Madmom for beat detection
- Continuous retraining on larger datasets

---

## 5. EDM-Specific Analysis

### Energy Features

From [Spotify audio analysis research](https://medium.com/data-science/genre-classification-of-electronic-dance-music-using-spotifys-audio-analysis-7350cac7daf0):

Energy (0.0-1.0) represents perceptual intensity/activity based on:
- Dynamic range
- Perceived loudness
- Timbre
- Onset rate
- General entropy

### EDM Subgenre Classification

[Deep learning EDM classification](https://arxiv.org/abs/2110.08862) using:
- Mel-spectrograms
- Fourier tempograms
- Autocorrelation tempograms

Key predictors: duration, loudness, tempo, danceability, energy

**Source**: [arxiv.org/html/2409.06690v3](https://arxiv.org/html/2409.06690v3) - Benchmarking Sub-Genre Classification for Mainstage Dance Music (2024)

---

## 6. Multi-Task Learning

### Joint Estimation Benefits

[2025 research](https://arxiv.org/html/2510.18190) on multi-task multi-scale networks shows:
- Joint estimation of dynamics, change points, beats, downbeats
- 60-second temporal context
- Only 0.5M parameters with MMoE decoder

### Multi-Task Self-Supervised Pre-training

[NSF research](https://par.nsf.gov/servlets/purl/10249858):
- Used ~100k AudioSet music clips (~83 hours)
- Multi-task pre-training improves downstream classification

---

## 7. Learning with Noisy Labels

### The Problem

From [KDnuggets](https://www.kdnuggets.com/2021/04/imerit-noisy-labels-impact-machine-learning.html) and [Google AI](https://ai.googleblog.com/2020/08/understanding-deep-learning-on.html):
- Label noise from crowd-sourcing, web scraping, automated systems is common
- Incorrect labels reduce model accuracy on clean test data
- Models may memorize labeling errors, causing unwanted biases

### Survey Paper

[arxiv.org/abs/2007.08199](https://arxiv.org/abs/2007.08199) - "Learning from Noisy Labels with Deep Neural Networks: A Survey"

Comprehensive coverage of methods for handling noisy labels.

### Cleanlab: Confident Learning

[Cleanlab](https://github.com/cleanlab/cleanlab) is the standard tool for detecting and handling label errors:

**Key capabilities**:
- Detect data issues (outliers, duplicates, label errors)
- Train robust models despite noisy labels
- Infer consensus + annotator quality for multi-annotator data
- Active learning: suggest data to (re)label next

**Technical approach**:
- Uses predicted probabilities from any ML model
- Estimates joint distribution of noisy and true labels
- No hyperparameters required
- Works with any ML framework (text, image, tabular, audio)

**Impact**: Found 100,000+ label errors in ImageNet

**Source**: [dcai.csail.mit.edu/2024/label-errors](https://dcai.csail.mit.edu/2024/label-errors)

---

## 8. Semi-Supervised Learning for Music

### Teacher-Student Models

[ISMIR 2020](https://program.ismir2020.net/static/final_papers/160.pdf) - Semi-supervised learning using teacher-student models:

1. Teacher model trained on labeled data
2. Student model learns to predict pseudo-labels on unlabeled data
3. Strong data augmentation on unlabeled data
4. Student can surpass teacher performance

### Noisy Student Training for Music

[arxiv.org/abs/2112.00702](https://arxiv.org/abs/2112.00702) - Semi-supervised music emotion recognition:
- Requires strong teacher model
- Benefits from complementary music representations

### Scaling MIR with Semi-Supervised Learning

[arxiv.org/abs/2310.01353](https://arxiv.org/abs/2310.01353) - Demonstrates scaling up MIR training with semi-supervised approaches.

---

## 9. Transfer Learning with Limited Data

### Key Findings

From [arxiv.org/html/2505.06042](https://arxiv.org/html/2505.06042) - "Learning Music Audio Representations With Limited Data":
- Under certain conditions, limited-data models perform comparably to large-dataset models
- Handcrafted features may outperform learned representations in some tasks

### YAMNet Transfer Learning

[TensorFlow tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio):
- Transfer from pre-trained audio models
- Achieves good results with few hundred samples for fine-tuning
- Data augmentation preserving semantic validity improves performance

### Practical Example

[Applied Sciences research](https://www.mdpi.com/2076-3417/11/24/11663/htm):
- Used 300 training samples
- Transfer from sound/music recognition to fault detection
- Maximum accuracy with few hundred fine-tuning samples

---

## 10. Training Data Requirements

### Minimum Viable Dataset Sizes

| Task | Minimum Samples | Notes |
|------|-----------------|-------|
| Transfer learning fine-tuning | 300-500 | With pretrained backbone |
| Beat tracking | ~2,000 files | Beat This! used 16 datasets |
| Cue point detection | ~4,700 tracks | EDM-CUE dataset |
| Multi-task pre-training | ~100k clips | ~83 hours of audio |
| Semi-supervised | 1:10 labeled:unlabeled | Teacher-student ratio |

### Quality vs. Quantity

From the research:
- **Clean subset critical**: Even small clean validation set enables confident learning
- **Noisy labels tolerable**: With proper techniques (cleanlab, noisy student)
- **Multi-task helps**: Joint learning improves all tasks with same data

---

## 11. Recommended Architecture Approaches

### Option A: Foundation Model + Fine-tuning

```
MERT/MuQ pretrained → Fine-tune heads for:
  - Energy regression
  - Segment boundary detection
  - Beat/downbeat tracking
```

**Pros**: Minimal training data, strong generalization
**Cons**: May not capture EDM-specific features

### Option B: All-In-One Style Multi-Task

```
Mel spectrogram → Shared encoder → Multi-task decoder:
  - Beat/downbeat head
  - Segment boundary head
  - Energy regression head
  - Key detection head
```

**Pros**: Tasks reinforce each other, EDM-specific
**Cons**: Needs more training data

### Option C: Hybrid (Recommended)

```
Phase 1: Fine-tune MERT on clean subset
Phase 2: Teacher-student on noisy data
Phase 3: Cleanlab to identify errors
Phase 4: Retrain on cleaned data
```

---

## 12. Key Resources

### Code Repositories
- [all-in-one](https://github.com/mir-aidj/all-in-one) - SOTA music structure analyzer
- [beat_this](https://github.com/CPJKU/beat_this) - SOTA beat tracking
- [cleanlab](https://github.com/cleanlab/cleanlab) - Label error detection
- [awesome-deep-learning-music](https://github.com/ybayle/awesome-deep-learning-music) - Curated list

### Datasets
- [EDM-CUE](https://arxiv.org/html/2407.06823v1) - 4,710 EDM tracks with cue points
- [Harmonix Set](https://github.com/urinieto/harmonixset) - Structure annotations
- [beat_this_annotations](https://github.com/CPJKU/beat_this_annotations) - Beat annotations

### Survey Papers
- [Learning from Noisy Labels](https://arxiv.org/abs/2007.08199)
- [Awesome Learning with Label Noise](https://github.com/subeeshvasu/Awesome-Learning-with-Label-Noise)

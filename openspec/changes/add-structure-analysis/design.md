# Design: Structure Analysis

## Context

EDM tracks follow predictable structural patterns (intro → buildup → drop → breakdown → drop → outro) that DJs rely on for mixing. Automated detection enables cue point generation, mix planning, and track categorization.

**Constraints:**
- Must run on consumer hardware (RTX 3060+ or CPU-only)
- < 30 seconds processing time
- Drop detection precision >90%
- Integrate with existing parallel processing infrastructure

**Stakeholders:**
- DJs wanting automated cue points
- Library users analyzing track collections
- Future DJ software integrations (Rekordbox export)

## Goals / Non-Goals

**Goals:**
- Detect core EDM sections: intro, buildup, drop, breakdown, outro
- Achieve >90% precision on drop detection
- Support both GPU and CPU inference
- Provide confidence scores for each section
- Enable accuracy evaluation against ground truth

**Non-Goals:**
- Sub-section detection (e.g., riser within buildup)
- Real-time streaming analysis
- Training custom models (future work)
- Genre-specific models (house vs techno vs trance)

## Decisions

### Decision 1: Use Allin1 as Primary Detector

**What:** Integrate the Allin1 music structure analysis model as the primary detection method.

**Why:**
- State-of-the-art transformer architecture (2023)
- Pre-trained on diverse music datasets
- Outputs section boundaries with labels
- Active maintenance, pip-installable
- Can be fine-tuned on EDM data later

**Alternatives considered:**
1. **MSAF (Music Structure Analysis Framework)** - Older algorithms, lower accuracy, but more interpretable
2. **Custom CNN+RNN** - Per project.md architecture, but requires training data and infrastructure
3. **Pure rule-based** - Simpler but significantly lower accuracy

**Trade-off:** Dependency on external model vs. training infrastructure. Chose external model for faster time-to-value.

### Decision 2: Energy-Based Fallback Detector

**What:** Implement librosa-based energy analysis as fallback when Allin1 unavailable.

**Why:**
- Allin1 has heavy dependencies (torch, transformers)
- Some users may not have GPU resources
- Energy-based drop detection well-established for EDM
- Provides baseline for accuracy comparison

**Implementation:**
```python
# Pseudocode for energy-based detection
def detect_drops_energy(audio, sr):
    rms = librosa.feature.rms(y=audio)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

    # Drops = high RMS + high bass contrast
    drop_candidates = find_peaks(rms * bass_weight)
    return filter_by_duration(drop_candidates, min_duration=8_bars)
```

### Decision 3: Section Label Mapping Strategy

**What:** Map generic Allin1 labels to EDM-specific terminology with context awareness.

**Mapping rules:**
1. `chorus` → `drop` (high energy sections)
2. `bridge` → `breakdown` (low energy melodic sections)
3. `verse` → `buildup` (pre-chorus tension)
4. Context refinement: consecutive "verse" before "chorus" = buildup

**Why not 1:1 mapping:**
- EDM doesn't have "verses" in the lyrical sense
- "Buildup" is kinetic (rising energy), not static
- Breakdown vs bridge distinction matters for DJs

### Decision 4: Detector Interface Pattern

**What:** Use strategy pattern for detector implementations, consistent with existing BPM detection.

```python
class StructureDetector(Protocol):
    def detect(self, audio: AudioData) -> list[Section]: ...

class Allin1Detector:
    def detect(self, audio: AudioData) -> list[Section]: ...

class EnergyDetector:
    def detect(self, audio: AudioData) -> list[Section]: ...

def analyze_structure(
    filepath: Path,
    *,
    detector: str = "auto"  # "allin1", "energy", "auto"
) -> StructureResult:
    ...
```

**Why:** Matches existing `compute_bpm()` pattern with `prefer_madmom` flag. Enables easy addition of new detectors.

### Decision 5: Integration with Parallel Processing

**What:** Structure analysis will use the existing parallel processing infrastructure.

**Considerations:**
- Allin1 model loading is expensive (~2GB VRAM)
- Should share model instance across workers
- CPU fallback when GPU unavailable or memory constrained

**Approach:**
- Load model once per worker process
- Use ProcessPoolExecutor with initializer for model loading
- Graceful fallback to energy detector if model load fails

## Risks / Trade-offs

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Allin1 accuracy poor on EDM | Medium | High | Energy fallback + future fine-tuning path |
| Model too slow on CPU | Low | Medium | Document GPU requirements, optimize batch size |
| Allin1 package breaks | Low | Medium | Pin version, energy fallback available |
| Memory pressure with parallel | Medium | Medium | Sequential mode option, model sharing |

## Migration Plan

1. **Phase 1:** Implement Allin1 detector, keep placeholder as temporary fallback
2. **Phase 2:** Add energy-based detector
3. **Phase 3:** Remove placeholder, integrate evaluation
4. **Rollback:** Revert to placeholder if critical issues found

No breaking API changes - `analyze_structure()` signature unchanged, just returns real data.

## Open Questions

1. **Fine-tuning data:** Where to source labeled EDM tracks for potential fine-tuning?
   - Potential: Manually label subset of test tracks
   - Potential: Parse Rekordbox cue points as weak labels

2. **Confidence calibration:** How to calibrate confidence scores across detectors?
   - Allin1 outputs logits that need calibration
   - Energy detector confidence based on energy delta magnitude

3. **Multi-drop handling:** How to distinguish multiple drops in extended mixes?
   - Current: Label all high-energy sections as "drop"
   - Future: Numbering (drop_1, drop_2) or sub-labeling

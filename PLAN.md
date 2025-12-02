# Improving Structure Analysis Accuracy

## Current System Analysis

### What the system does now

1. **Boundary Detection (via MSAF)**
   - Uses spectral flux (`sf`) algorithm to find where sections change
   - Returns raw boundary times (e.g., `[0, 32.5, 64.2, 128.7, ...]`)
   - Returns cluster IDs for segment similarity (e.g., `[0, 1, 2, 0, 1, ...]`)
   - MSAF's cluster IDs indicate *similarity* between segments, not semantic labels

2. **Energy-based Label Mapping**
   - Takes MSAF boundaries and applies heuristics:
     - First segment → `intro` (always)
     - Last segment → `outro` (always)
     - Energy > 0.7 → `drop`
     - Energy gradient > 0.15 → `buildup`
     - Energy < 0.4 → `breakdown`
     - Otherwise → `other`

### Current Weaknesses

1. **Positional bias**: First/last segments are hard-coded as intro/outro regardless of actual content
2. **Single feature (energy)**: Only uses RMS energy for classification
3. **Fixed thresholds**: 0.7, 0.4, 0.15 are arbitrary, not learned from data
4. **No temporal context**: Each segment classified independently
5. **MSAF clusters ignored**: The similarity information from MSAF's labeling algorithm is discarded

---

## Key Insight: Two Separate Problems

### Problem 1: Boundary Detection
- **Question**: *Where* do sections change?
- **Current approach**: MSAF spectral flux (works reasonably well)
- **Ground truth needed**: Precise timestamps where sections change
- **Metric**: Boundary F1 (within ±2 second tolerance)

### Problem 2: Label Classification
- **Question**: *What type* is each section?
- **Current approach**: Energy thresholds (poor)
- **Ground truth needed**: Label for each detected segment
- **Metric**: Label accuracy (% correct given correct boundaries)

---

## What Data Would Help Most

### High-Value Data (in priority order)

#### 1. Labeled Structure Annotations (MOST VALUABLE)
You already have one example (`data/dj-rhythmcore-narcos_annotated.yaml`). More of these is the single highest-impact data you can provide.

**Format (you're already using):**
```yaml
file: /path/to/track.flac
duration: 242.2
bpm: 128.0
source: manual_annotation
structure:
- [1, 24, intro]
- [25, 80, drop]       # vs. tool's 'other'
- [81, 92, breakdown]
- [93, 104, buildup]
- [105, 126, drop]
- [127, 130, outro]
```

**What makes a good annotation:**
- Bar boundaries that align to typical EDM phrasing (4, 8, 16, 32 bars)
- Consistent labeling philosophy (what's a "drop" vs "breakdown"?)
- Diverse genres/sub-genres (house, dubstep, trance, techno, etc.)

#### 2. Boundary-Only Annotations (MEDIUM VALUE)
If labeling is tedious, just marking *where* sections change is still valuable:

```yaml
file: /path/to/track.flac
bpm: 128.0
boundaries: [1, 17, 33, 65, 81, 97, 113, 128]  # bar numbers
```

This helps evaluate/improve boundary detection without requiring labels.

#### 3. Drop Timestamps (EASY TO CREATE)
If you can only spare minimal effort, just marking drop starts is useful:

```csv
filename,bar,label,bpm
track.flac,33,drop,128
track.flac,97,drop,128
```

---

## Next Steps for Improvement

### Option A: Improve Heuristics (Quick Win)

**What to do:**
1. Analyze the discrepancy between your annotation and tool output
2. Identify what audio features differ between sections you labeled `drop` vs the tool's `other`
3. Tune thresholds or add features

**Data needed:** 5-10 annotated tracks to find patterns

**Expected improvement:** Moderate (maybe +10-20% label accuracy)

### Option B: Learn Thresholds from Data (Medium Effort)

**What to do:**
1. Collect annotations for 20-30 tracks
2. Extract features (energy, spectral centroid, onset density, bass energy, etc.)
3. Learn optimal thresholds/weights via logistic regression or decision tree

**Data needed:** 20-30 annotated tracks

**Expected improvement:** Good (+20-40% label accuracy)

### Option C: Retrain/Fine-tune MSAF (High Effort)

**What to do:**
1. Collect annotations for 50+ tracks
2. Use MSAF's training framework to tune boundary detection
3. Or replace with a modern deep learning approach (requires more data)

**Data needed:** 50+ tracks with precise boundaries

**Expected improvement:** Potentially large, but high effort

---

## Recommended Action

### Immediate: Create a small evaluation set

Annotate 5-10 diverse tracks with full structure labels. This enables:
1. Measuring current accuracy quantitatively
2. Identifying systematic errors (e.g., "always misclassifies X as Y")
3. Tuning heuristics based on real data

### Data Collection Strategy

1. **Select diverse tracks**: Different sub-genres, tempos, structures
2. **Use your existing format**: The YAML format you have works well
3. **Focus on sections that matter**: intro, buildup, drop, breakdown, outro
4. **Accept "other"**: Some sections don't fit neatly - label them `other`

### Annotation Workflow

```bash
# 1. Run tool on a track
uv run edm analyze structure /path/to/track.flac --output yaml > track_tool.yaml

# 2. Copy and edit
cp track_tool.yaml track_annotated.yaml
# Listen to track, correct start_bar/end_bar/label for each section

# 3. Run evaluation (once you have reference CSV)
uv run edm evaluate structure --source /path/to/music --reference annotations.csv
```

---

## Summary

| Data Type | Effort | Impact | Quantity Needed |
|-----------|--------|--------|-----------------|
| Full structure annotations | High | Very High | 5-30 tracks |
| Boundary-only annotations | Medium | Medium | 10-50 tracks |
| Drop timestamps only | Low | Low-Medium | 20-100 tracks |

**Recommendation**: Start with 5-10 full structure annotations. This small dataset will reveal where the model fails and guide whether you need better features, more data, or a different approach entirely.

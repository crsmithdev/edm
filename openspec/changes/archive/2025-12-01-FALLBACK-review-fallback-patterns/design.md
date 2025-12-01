# Design: Fallback Pattern Review

## Inventory of Fallback Patterns

### 1. Beat Detection (beat_detector.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| beat_this → librosa | ImportError | librosa implementation | **Keep** - beat_this is GPU-optional |
| CUDA → CPU | No GPU | CPU inference | **Keep** - legitimate optimization |

### 2. BPM Detection (bpm_detector.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| beat_this → librosa | ImportError or failure | librosa implementation | **Keep** - beat_this is GPU-optional |
| prefer_madmom flag | User preference | Reverse order | **Keep** - user control |

### 3. BPM Analysis (bpm.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| metadata → computed | No BPM tag | Compute from audio | **Keep** - workflow step, not fallback |
| ignore_metadata flag | User preference | Skip metadata | **Keep** - user control |

Note: The metadata → computed flow is a normal workflow step (check tags first, compute if missing), not a fallback pattern. No code changes needed, just ensure naming/docs don't call it "fallback".

### 4. Structure Detection (structure_detector.py, structure.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| MSAFDetector.is_available() | ImportError | Returns False | **Remove** - make msaf required |
| MSAF → EnergyDetector | Detection failure | Energy-based | **Remove** - fail loudly, fix issues |
| get_detector() auto mode | MSAF unavailable | EnergyDetector | **Remove** - make msaf required |

### 5. Visualization (evaluation/common.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| matplotlib import | ImportError | Skip plot | **Keep** - only used for benchmarks |

### 6. Metadata (metadata.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| Title tag → filename | No tag | Use filename | **Keep** - sensible default |

### 7. Git Info (evaluation/common.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| git commit/branch | No git | Return "unknown" | **Keep** - handles edge case |

### 8. Audio Caching (audio.py)
| Pattern | Trigger | Fallback | Recommendation |
|---------|---------|----------|----------------|
| Cache disabled | max_size=0 | Skip caching | **Keep** - feature toggle |

## Recommendations Summary

### Make Required (remove fallback)
1. **msaf** - Primary structure detector, always want best results

### Keep Conditional Imports (test/benchmark only)
1. **matplotlib** - Only used for benchmark plots

### Keep Fallbacks
1. **beat_this → librosa** - Optional GPU acceleration, librosa is fine CPU fallback
2. **CUDA → CPU** - Auto-detect GPU availability

### Not Really Fallbacks (keep as-is)
1. **metadata → computed BPM** - Workflow step
2. **Title → filename** - Sensible default
3. **Git unavailable** - Edge case handling
4. **Cache disabled** - Feature toggle

### Remove
1. **MSAF → EnergyDetector runtime fallback** - Fail loudly, don't silently degrade
2. **is_available() pattern** - Direct import, fail fast if msaf missing
3. **get_detector() auto mode** - No fallback needed, msaf is required

## Architecture Notes

### Current Conditional Import Pattern
```python
def is_available(self) -> bool:
    try:
        import msaf
        self._msaf = msaf
        return True
    except ImportError:
        return False
```

### Proposed Direct Import Pattern
```python
import msaf  # Top-level, fail fast if missing

class MSAFDetector:
    def detect(self, ...):
        # No availability check needed
        boundaries, labels = msaf.process(...)
```

## MSAF Runtime Errors

MSAF can fail at runtime even when installed. Known error scenarios:

1. **JSON cache corruption** - Fixed by using per-process cache files (commit 1df0f66)
2. **scipy.inf deprecation** - Already patched in is_available() (lines 82-87)
3. **Algorithm failures** - Some audio files may cause numerical errors in segmentation
4. **Memory errors** - Very long files may exhaust memory during feature extraction
5. **librosa/numpy version conflicts** - MSAF depends on specific versions

**Decision**: Remove the MSAF → EnergyDetector fallback. If MSAF fails, surface the error loudly so it can be fixed rather than silently degrading to lower-quality results. Users should know when something is broken.

## Other Patterns Explained

### Title → Filename (metadata.py:97)
When audio file has no title tag, use the filename stem as title. This is a sensible default, not a fallback - every file has a name.

### Git Info (evaluation/common.py:152-166)
Evaluation reports include git commit/branch for reproducibility. If git unavailable (e.g., running from tarball), returns "unknown". Low-impact edge case handling.

### Cache Toggle (audio.py:47-48)
Audio cache can be disabled by setting max_size=0. This is a feature flag, not a fallback - allows users to trade memory for disk I/O.

## Trade-offs

### Making msaf Required
- **Pro**: Simpler code, better error messages, clearer requirements
- **Pro**: No need for EnergyDetector as fallback for missing deps
- **Con**: Users must install msaf to use any structure analysis
- **Mitigation**: Good error message if missing, document in README

### Keeping beat_this Optional
- **Pro**: Works on CPU-only systems with librosa
- **Pro**: GPU is a nice-to-have, not essential
- **Con**: Maintains fallback complexity
- **Decision**: Keep - GPU acceleration is genuinely optional

## Requirements

1. Update pyproject.toml to make msaf, matplotlib non-optional
2. Remove is_available() pattern for required dependencies
3. Keep fallback patterns for genuinely optional features
4. Improve error messages for missing dependencies
5. Update documentation to reflect requirements

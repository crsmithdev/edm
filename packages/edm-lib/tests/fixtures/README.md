# Test Fixtures

This directory contains test audio files for the EDM Analyzer test suite.

## Synthetic Audio Files

Following best practices from `librosa` and `madmom` test suites, we use synthetic audio files with known ground truth BPM values for unit testing.

### Generated Files

Run `python generate_test_audio.py` to create:

- **Click tracks** (`click_XXXbpm.wav`): Simple impulse trains at exact beat positions
  - Used for testing core BPM detection algorithms
  - Provides known ground truth for validation

- **Beat patterns** (`beat_XXXbpm.wav`): Realistic drum patterns with kick, snare, hi-hat
  - Tests algorithm robustness to complex audio
  - More realistic than click tracks

### BPMs Covered

- 120 BPM - House
- 125 BPM - Tech House
- 128 BPM - Standard EDM
- 140 BPM - Dubstep / Trap
- 150 BPM - Hardstyle
- 174 BPM - Drum & Bass

All files are:
- 10 seconds duration
- 44.1 kHz sample rate
- Stereo
- WAV format

## Test Tolerance

Following industry standards, BPM detection tests allow ±5% tolerance from ground truth values.

## References

This approach follows established practices from:
- librosa: https://github.com/librosa/librosa/tree/main/tests
- madmom: https://github.com/CPJKU/madmom/tree/main/tests

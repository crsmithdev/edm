## 1. Beat Grid Data Types

- [x] 1.1 Create `BeatGrid` dataclass with `first_beat_time`, `bpm`, `time_signature`, `confidence`, `method`
- [x] 1.2 Implement `BeatGrid.first_downbeat` property
- [x] 1.3 Implement `BeatGrid.beat_to_time(beat_index)` method
- [x] 1.4 Implement `BeatGrid.time_to_beat(time)` method
- [x] 1.5 Implement `BeatGrid.to_beat_times(duration)` method
- [x] 1.6 Implement `BeatGrid.to_downbeat_times(duration)` method
- [x] 1.7 Add unit tests for `BeatGrid`

## 2. Beat Grid from Timestamps

- [x] 2.1 Implement `BeatGrid.from_timestamps(beats, downbeats)` class method
- [x] 2.2 Add unit tests for timestamp conversion

## 3. Beat Detection Functions

- [x] 3.1 Create `src/edm/analysis/beat_detector.py` module
- [x] 3.2 Implement `detect_beats()` using beat_this, returns `BeatGrid`
- [x] 3.3 Implement `detect_beats_librosa()` fallback
- [x] 3.4 Add unit tests for beat detection functions

## 4. Bar Calculation Updates

- [x] 4.1 Add `time_to_bar()` method to `BeatGrid`
- [x] 4.2 Add `bar_to_time()` method to `BeatGrid`
- [x] 4.3 Add unit tests for bar calculations with `BeatGrid`

## 5. Annotation Support

- [x] 5.1 Add `first_downbeat` column support to CSV annotation loader
- [x] 5.2 Update bar calculations in evaluation to use per-track first_downbeat
- [x] 5.3 Add unit tests for first_downbeat annotation parsing

## 6. CLI Updates

- [x] 6.1 Add `edm analyze beats` command or integrate into existing analyze
- [x] 6.2 Display beat grid parameters (first_beat_time, bpm, time_signature)

## 7. Integration Testing

- [x] 7.1 Integration test with real audio file
- [x] 7.2 Verify beat_this → BeatGrid conversion

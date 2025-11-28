# Structure Annotations - Bar-Based Format

This document contains the automatically detected structure for 5 EDM tracks, expressed in **bars** rather than seconds for easier editing while listening.

**IMPORTANT:** Bar numbering is 1-indexed to match DJ software conventions (Rekordbox, Traktor, Ableton, Serato). Bar 1 is the first bar of the track, and an 8-bar intro spans bars 1-8 (inclusive).

## Detected Structure Summary

### Track 1: 3LAU, Dnmo - Falling.flac
- **BPM:** 132.0
- **Total bars:** ~116 bars (209.1 seconds)
- **Sections detected:** 15

| Start Bar | End Bar | Bars | Label | Notes |
|-----------|---------|------|-------|-------|
| 1 | 2 | 1 | intro | Very short intro |
| 2 | 7 | 5 | drop | Early drop |
| 7 | 17 | 10 | breakdown |  |
| 17 | 25 | 8 | buildup |  |
| 25 | 34 | 9 | drop |  |
| 34 | 42 | 8 | drop |  |
| 42 | 50 | 8 | drop |  |
| 50 | 58 | 8 | drop |  |
| 58 | 66 | 8 | drop |  |
| 66 | 74 | 8 | breakdown |  |
| 74 | 81 | 7 | buildup |  |
| 81 | 89 | 8 | breakdown |  |
| 89 | 98 | 9 | drop |  |
| 98 | 116 | 18 | drop |  |
| 116 | 116 | 0 | outro |  |

### Track 2: ANDATA Teenage Mutants - Black Milk (Teenage Mutants Remix).flac
- **BPM:** 131.0
- **Total bars:** ~225 bars (410.4 seconds)
- **Sections detected:** 14

| Start Bar | End Bar | Bars | Label | Notes |
|-----------|---------|------|-------|-------|
| 1 | 1 | 0 | intro | Very short intro |
| 1 | 5 | 4 | breakdown |  |
| 5 | 67 | 62 | drop | Very long drop |
| 67 | 76 | 9 | breakdown |  |
| 76 | 118 | 42 | breakdown | Very long breakdown |
| 118 | 124 | 6 | buildup |  |
| 124 | 161 | 37 | drop | Very long drop |
| 161 | 172 | 11 | drop |  |
| 172 | 182 | 10 | drop |  |
| 182 | 189 | 7 | breakdown |  |
| 189 | 203 | 14 | drop |  |
| 203 | 212 | 9 | drop |  |
| 212 | 225 | 13 | breakdown |  |
| 225 | 225 | 0 | outro |  |

### Track 3: AUTOFLOWER - Dimension.flac
- **BPM:** 130.0
- **Total bars:** ~127 bars (232.6 seconds)
- **Sections detected:** 8

| Start Bar | End Bar | Bars | Label | Notes |
|-----------|---------|------|-------|-------|
| 1 | 1 | 0 | intro | Very short intro |
| 1 | 46 | 45 | drop | Very long drop |
| 46 | 51 | 5 | drop |  |
| 51 | 59 | 8 | breakdown |  |
| 59 | 70 | 11 | breakdown |  |
| 70 | 95 | 25 | drop | Long drop |
| 95 | 127 | 32 | drop | Long drop |
| 127 | 127 | 0 | outro |  |

### Track 4: AUTOFLOWER - THE ONLY ONE.flac
- **BPM:** 126.0
- **Total bars:** ~115 bars (217.1 seconds)
- **Sections detected:** 8

| Start Bar | End Bar | Bars | Label | Notes |
|-----------|---------|------|-------|-------|
| 1 | 2 | 1 | intro | Very short intro |
| 2 | 8 | 6 | drop | Early drop |
| 8 | 34 | 26 | drop | Long drop |
| 34 | 61 | 27 | drop | Long drop |
| 61 | 99 | 38 | drop | Very long drop |
| 99 | 106 | 7 | drop |  |
| 106 | 115 | 9 | drop |  |
| 115 | 115 | 0 | outro |  |

### Track 5: AUTOFLOWER - Wallflower.flac
- **BPM:** 124.0
- **Total bars:** ~131 bars (251.1 seconds)
- **Sections detected:** 17

| Start Bar | End Bar | Bars | Label | Notes |
|-----------|---------|------|-------|-------|
| 1 | 1 | 0 | intro | Very short intro |
| 1 | 8 | 7 | buildup |  |
| 8 | 17 | 9 | drop | Early drop |
| 17 | 24 | 7 | breakdown |  |
| 24 | 42 | 18 | drop |  |
| 42 | 50 | 8 | drop |  |
| 50 | 58 | 8 | drop |  |
| 58 | 66 | 8 | breakdown |  |
| 66 | 72 | 6 | breakdown |  |
| 72 | 81 | 9 | buildup |  |
| 81 | 90 | 9 | breakdown |  |
| 90 | 98 | 8 | drop |  |
| 98 | 105 | 7 | drop |  |
| 105 | 114 | 9 | drop |  |
| 114 | 121 | 7 | drop |  |
| 121 | 131 | 10 | drop |  |
| 131 | 131 | 0 | outro |  |

## Editing Instructions

The bar-based annotations are saved in:
**`tests/fixtures/reference/structure_annotations_bars_5files.csv`**

### How to Edit

1. **Open in your music player/DJ software** - Load each track and enable bar display
2. **Listen and mark sections** - Note the actual bar numbers where sections change
3. **Edit the CSV** - Update `start_bar` and `end_bar` columns with correct values
4. **Keep BPM accurate** - The BPM values are auto-detected, verify they're correct

### CSV Format

```csv
filename,start_bar,end_bar,label,bpm
track.flac,1,17,intro,128
track.flac,17,33,buildup,128
track.flac,33,65,drop,128
```

**Note:** Bar numbering is 1-indexed. An 8-bar intro spans bars 1-8, and the following section starts at bar 9.

### Section Labels

- **intro** - Opening section
- **buildup** - Rising energy, tension building before drop
- **drop** - High-energy payoff section (main hook, bass drop)
- **breakdown** - Reduced energy, melodic focus, stripped-back
- **outro** - Closing section

### Tips

- **Typical EDM structure**: intro (8-16 bars) → buildup (8-16 bars) → drop (16-32 bars) → breakdown (8-16 bars) → buildup → drop → outro
- **Bar multiples**: Sections typically align to 4, 8, 16, or 32 bar phrases
- **Player display**: Most DJ software shows bars in format like "64.1.1" (bar 64, beat 1, sub-beat 1)
- **Start of drop**: Usually marked by bass return, energy increase, main synth/vocal hook
- **Bar 1 convention**: The first bar of the track is bar 1, not bar 0

## Evaluation

Once edited, evaluate the accuracy:

```bash
uv run edm evaluate structure \
  --source ~/music \
  --reference tests/fixtures/reference/structure_annotations_bars_5files.csv \
  --full
```

This will compare the auto-detected structure against your corrected annotations and report precision/recall metrics.

# EDM Annotator Frontend Architecture

Modern React + TypeScript frontend for annotating EDM track structure.

## Overview

The frontend uses a dual-waveform display system with Zustand for state management, providing an intuitive interface for marking structure boundaries and regions in EDM tracks.

## Dual Waveform System

### OverviewWaveform
- **Purpose**: Full track visualization with moving playhead
- **Height**: 60px compact view
- **Waveform**: Non-mirrored, extends upward from baseline
- **Playhead**: Moves left-to-right as track plays
- **Interaction**: Click to seek
  - With quantize enabled: Snaps to nearest bar
  - With quantize disabled: Seeks to exact click position
- **Visual elements**:
  - Subtle region overlays (colored, 0.15 opacity)
  - Boundary markers (purple, 2px, 0.7 opacity)
  - Unlabeled regions show no overlay

### DetailWaveform
- **Purpose**: Detailed editing view with centered playhead
- **Waveform**: 3-band mirrored display (bass/mids/highs)
- **Playhead**: Fixed at center (50%), waveform scrolls underneath
- **Viewport**: Shows configurable time span (4-60s, default 16s)
- **Empty space**: Shows at track start/end when playhead near boundaries
- **Scaling**: Uses global max amplitude for consistent height across all viewport positions

### Interaction Patterns

**Mouse interactions:**
- **Drag (regular)**: Scrubs playback position
  - Drag left: Move forward in time (waveform moves left)
  - Drag right: Move backward in time (waveform moves right)
  - Respects quantize: snaps to nearest beat when enabled
- **Shift + Drag**: Scrubs with quantize bypass (fine control)
- **Ctrl/Cmd + Click**: Adds boundary at clicked position
  - Snaps to nearest beat when quantize enabled
  - Exact position when quantize disabled

**Cursors:**
- Overview: Default arrow cursor
- Detail: Grab cursor (hand) by default
- Detail with Ctrl held: Crosshair (boundary mode)
- Detail while dragging: Grabbing cursor

## State Management (Zustand)

### Store Architecture

Six focused stores provide clean separation of concerns:

#### audioStore
- **Responsibility**: Playback state and control
- **State**: `player`, `isPlaying`, `currentTime`, `cuePoint`
- **Actions**: `play()`, `pause()`, `seek()`, `setCuePoint()`, `returnToCue()`
- **Cue point**: Snaps to beat when quantize enabled

#### waveformStore
- **Responsibility**: Waveform data and viewport
- **State**: Waveform arrays (bass/mids/highs/times), duration, viewport bounds, zoom level
- **Actions**: `setWaveformData()`, `setViewport()`, `zoom()`, `pan()`
- **Viewport**: DetailWaveform passes actual viewport (can be negative) to overlays via props

#### structureStore
- **Responsibility**: Boundaries and regions
- **State**: `boundaries` (array of times), `regions` (array of {start, end, label})
- **Actions**: `addBoundary()`, `removeBoundary()`, `setRegionLabel()`
- **Label types**: intro, buildup, main, breakdown, outro, unlabeled

#### tempoStore
- **Responsibility**: BPM, downbeat, time/bar conversions
- **State**: `trackBPM`, `trackDownbeat`
- **Actions**: `setBPM()`, `setDownbeat()`, `tapTempo()`
- **Utilities**: Time-to-bar and bar-to-time conversions

#### uiStore
- **Responsibility**: UI state flags
- **State**: `quantizeEnabled`, `isDragging`, `jumpMode`, `statusMessage`
- **Actions**: `toggleQuantize()`, `showStatus()`, `setDragging()`

#### trackStore
- **Responsibility**: Track selection and loading
- **State**: `tracks`, `currentTrack`, `selectedTrack`
- **Actions**: `loadTrack()`, `selectTrack()`, `setTracks()`

### State Flow Examples

**Loading a track:**
1. User clicks track in TrackSelector
2. `trackService.loadTrack(filename)` fetches data
3. Updates `waveformStore` with waveform arrays
4. Updates `tempoStore` with BPM/downbeat
5. Updates `audioStore` with audio URL
6. Updates `structureStore` with boundaries/regions

**Setting cue point (C key):**
1. `useKeyboardShortcuts` detects 'C' key
2. Checks `audioStore.isPlaying`
3. If stopped:
   - Gets `currentTime` from `audioStore`
   - If quantize enabled: calculates nearest beat using `tempoStore`
   - Calls `audioStore.setCuePoint(time)`
4. If playing: calls `audioStore.returnToCue()`

**Drag-to-scrub:**
1. `DetailWaveform` detects mousedown (not Ctrl/Shift)
2. Records drag start position and time
3. On mousemove:
   - Calculates pixel delta → time delta
   - If quantize enabled AND shift not held: snaps to beat
   - Calls `audioStore.seek(newTime)`

## Key Components

### Waveform Components

#### WaveformContainer
- Container managing both waveforms
- Controls detail view span (zoom +/- buttons)
- Renders `OverviewWaveform` and `DetailWaveform`

#### DetailWaveform
- 3-band SVG waveform (bass cyan, mids purple, highs pink)
- Fixed center playhead
- Integrates overlays: `RegionOverlays`, `BeatGrid`, `BoundaryMarkers`
- Handles click/drag interactions

#### OverviewWaveform
- Single-color downsampled waveform (500 samples max)
- Moving playhead indicator
- Click-to-seek with bar snapping

### Overlay Components

#### BeatGrid
- Vertical lines for bars and beats
- **Downbeat**: Red, 3px
- **Bars**: Grey (rgba(200,200,200,0.7)), 2px, with bar numbers
- **Beats**: Lighter grey (rgba(150,150,150,0.3)), 1px
- Adaptive density: hides when zoomed out
- Bar numbers positioned 4px to right of line

#### BoundaryMarkers
- Purple (#7b6aff) vertical lines, 3px width
- Hover tooltip shows time
- Ctrl+click to delete
- Only visible when within viewport

#### RegionOverlays
- Colored semi-transparent backgrounds (0.4 opacity)
- Colors from `labelColors` mapping
- **Filters out unlabeled regions** - no overlay shown
- Clipped to viewport bounds

### Control Components

#### PlaybackControls
- Play/pause toggle
- Previous/next track buttons
- Display current time and duration

#### EditingControls
- "Boundary" button (adds at playhead)
- "Downbeat" button (sets at playhead)
- "Quantize" toggle
- Uses Button component with primary/secondary variants

#### RegionList
- Table of regions with start/end times
- Label dropdown for each region
- Delete button per region
- Click region to seek to start

#### TrackSelector
- Sidebar list of available tracks
- Shows annotation status (reference/generated)
- Click to load track

### Custom Hooks

#### useKeyboardShortcuts
- Handles all keyboard shortcuts
- Space: play/pause
- C: set cue (stopped) / return to cue (playing)
- R: return to cue
- B: add boundary at playhead
- D: set downbeat
- Q: toggle quantize
- Arrows: navigate by bars
- +/-: zoom
- Ignores when typing in inputs

## Visual Design

### Color Scheme

**Waveform:**
- Bass: `rgba(0, 229, 204, 0.8)` (cyan)
- Mids: `rgba(123, 106, 255, 0.8)` (purple)
- Highs: `rgba(255, 107, 181, 0.8)` (pink)

**Indicators:**
- Playhead: `#1affef` → `#00e5cc` gradient (cyan)
- Cue point: `#ff9500` → `#ff6b00` gradient (orange)
- Boundaries: `#7b6aff` (purple, matches button color)
- Downbeat: `#f44336` (red)
- Bar lines: `rgba(200, 200, 200, 0.7)` (grey)
- Beat lines: `rgba(150, 150, 150, 0.3)` (lighter grey)

**Regions:**
- intro: `#7b6aff` (purple)
- buildup: `#ff9800` (orange)
- main: `#4caf50` (green)
- breakdown: `#2196f3` (blue)
- outro: `#9c27b0` (purple)
- unlabeled: No overlay shown

### Backgrounds
- Waveform background: `#0a0a12` (dark)
- Container: `var(--bg-tertiary)`
- Borders: `var(--border-subtle)`

## Technical Details

### Viewport Synchronization

**Problem**: DetailWaveform viewport can extend beyond track bounds (negative start when at track beginning).

**Solution**:
1. DetailWaveform calculates unclamped viewport: `currentTime ± span/2`
2. Passes actual viewport to overlays via props: `viewportStart={viewport.start} viewportEnd={viewport.end}`
3. Overlay components use passed props instead of store values
4. Store viewport is clamped for other consumers

### Waveform Scaling

Uses `globalMaxAmplitude` calculated once from entire track:
```typescript
const globalMaxAmplitude = useMemo(() => {
  return Math.max(
    ...waveformBass.map(Math.abs),
    ...waveformMids.map(Math.abs),
    ...waveformHighs.map(Math.abs),
    0.001
  );
}, [waveformBass, waveformMids, waveformHighs]);
```

This prevents visual scaling artifacts as different parts of the track scroll into view.

### Quantize Implementation

Beat snapping using tempo information:
```typescript
const beatDuration = getBeatDuration(trackBPM);
const beatsFromDownbeat = (time - trackDownbeat) / beatDuration;
const nearestBeat = Math.round(beatsFromDownbeat);
const snappedTime = trackDownbeat + nearestBeat * beatDuration;
```

Applied to:
- Boundary placement (B key, Ctrl+click)
- Cue point setting (C key when stopped)
- Drag-to-scrub (unless Shift held)

Bar snapping (overview waveform only):
```typescript
const bar = timeToBar(rawTime, trackBPM, trackDownbeat);
const nearestBar = Math.round(bar);
const snappedTime = barToTime(nearestBar, trackBPM, trackDownbeat);
```

## Performance Considerations

- **Downsampling**: OverviewWaveform uses max 500 samples
- **Viewport culling**: DetailWaveform only processes visible samples
- **Memoization**: Heavy calculations wrapped in `useMemo`
- **Callbacks**: Event handlers wrapped in `useCallback`
- **Store granularity**: Multiple stores prevent unnecessary re-renders

## Future Enhancements

Potential areas for expansion:
- Zoom gestures (pinch-to-zoom on trackpad)
- Waveform color customization
- Multiple cue points
- Undo/redo for annotations
- Real-time collaboration
- Audio effects/filters

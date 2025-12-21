# EDM Annotator Frontend

Modern React + TypeScript frontend for the EDM Structure Annotator.

## ✅ Completed Infrastructure

### Core Setup
- ✅ Vite + React 18 + TypeScript configuration
- ✅ Package.json with all dependencies
- ✅ TypeScript strict mode configuration
- ✅ Vite proxy for backend API

### State Management (Zustand)
- ✅ **audioStore** - Playback state (player, isPlaying, currentTime, cuePoint)
- ✅ **trackStore** - Track selection and loading
- ✅ **waveformStore** - Waveform data and viewport (zoom, pan)
- ✅ **structureStore** - Boundaries and regions
- ✅ **tempoStore** - BPM, downbeat, tempo utilities
- ✅ **uiStore** - UI state (dragging, quantize, jump mode, status)

### Types & Interfaces
- ✅ Track types (Track, LoadedTrack)
- ✅ Waveform types (WaveformData)
- ✅ Structure types (Region, Boundary, SectionLabel)
- ✅ API types (request/response interfaces)

### Utilities
- ✅ **timeFormat** - Format seconds as MM:SS.mmm
- ✅ **barCalculations** - Convert between time/bars, calculate durations
- ✅ **quantization** - Snap to beat/bar boundaries
- ✅ **colors** - Label color mappings

### API Service
- ✅ Axios-based API client
- ✅ Track listing, loading, saving endpoints

## 🚧 Components to Implement

### Priority 1: Core Visualization

#### components/Waveform/WaveformCanvas.tsx
**Purpose**: Main waveform visualization with 3-band display

**Implementation Guide**:
```typescript
import { useWaveformStore } from "@/stores";

// Key features:
// - SVG-based rendering for scalability
// - 3 horizontal bands (bass/mids/highs)
// - Viewport culling (only render visible samples)
// - Adaptive detail: bars when zoomed out, continuous path when zoomed in
// - Click to set cue point, Shift+Click to add boundary

// Rendering strategy:
// 1. Calculate visible sample range from viewportStart/viewportEnd
// 2. For each band, map RMS values to SVG path
// 3. Switch from individual bars to <path> when pixelsPerSample < 2
// 4. Handle mouse events for cue point and boundary placement
```

#### components/Waveform/BeatGrid.tsx
**Purpose**: Overlay bar/beat grid lines on waveform

**Implementation Guide**:
```typescript
import { useTempoStore, useWaveformStore } from "@/stores";

// Key features:
// - Draw vertical lines for bars (orange, thick) and beats (gray, thin)
// - Downbeat line (red, thickest)
// - Bar numbers at top
// - Adaptive density: hide grid when too dense (pixelsPerBeat < 5)

// Calculation:
// 1. Get BPM and downbeat from tempoStore
// 2. Calculate bar/beat times within viewport
// 3. Convert times to pixel positions
// 4. Render as absolutely positioned divs or SVG lines
```

#### components/Waveform/Playhead.tsx
**Purpose**: Visual indicator of current playback position

**Implementation Guide**:
```typescript
import { useAudioStore, useWaveformStore } from "@/stores";

// Key features:
// - Vertical line following currentTime
// - Only visible when within viewport
// - Updates at 60fps during playback

// Implementation:
// - Use requestAnimationFrame for smooth updates
// - Calculate pixel position from currentTime and viewport
```

#### components/Waveform/BoundaryMarkers.tsx
**Purpose**: Draggable vertical markers for structure boundaries

**Implementation Guide**:
```typescript
import { useStructureStore, useWaveformStore } from "@/stores";

// Key features:
// - Vertical line for each boundary
// - Clickable to select/remove
// - Show time label on hover
// - Draggable to adjust position

// Interaction:
// - Click: select boundary
// - Drag: update boundary time
// - Right-click: remove boundary
```

#### components/Waveform/RegionOverlays.tsx
**Purpose**: Colored background regions showing structure sections

**Implementation Guide**:
```typescript
import { useStructureStore, useWaveformStore } from "@/stores";
import { labelColors } from "@/utils/colors";

// Key features:
// - Semi-transparent colored rectangles for each region
// - Click to select region (for label editing)
// - Highlight selected region

// Rendering:
// - Map each region to a div with calculated width/position
// - Apply color based on label
// - Handle click to update selected region in state
```

### Priority 2: Playback & Navigation

#### components/Transport/PlaybackControls.tsx
**Purpose**: Play/pause/cue buttons

**Implementation Guide**:
```typescript
import { useAudioStore } from "@/stores";

// Buttons:
// - Play/Pause toggle
// - Return to Cue
// - Previous Track / Next Track
// - Set Cue button

// Connect to audioStore actions:
// - play(), pause(), returnToCue()
```

#### components/Transport/NavigationControls.tsx
**Purpose**: Bar/beat jump buttons

**Implementation Guide**:
```typescript
import { useAudioStore, useTempoStore, useUIStore } from "@/stores";

// Buttons:
// - Jump -8 bars, -4 bars, -1 bar, +1 bar, +4 bars, +8 bars
// - Or jump by beats (depends on uiStore.jumpMode)
// - Mode toggle button

// Logic:
// - Calculate new time using barToTime() or beat duration
// - Call audioStore.seek()
```

### Priority 3: Editing Interface

#### components/Editing/EditingControls.tsx
**Purpose**: Annotation editing controls

**Implementation Guide**:
```typescript
import { useStructureStore, useTempoStore, useUIStore, useAudioStore } from "@/stores";

// Controls:
// - Add Boundary (at playhead)
// - Set Downbeat (at playhead)
// - Quantize toggle
// - Tempo input (BPM)
// - Tap Tempo button
// - Save button

// Actions:
// - structureStore.addBoundary(audioStore.currentTime)
// - tempoStore.setDownbeat(audioStore.currentTime)
// - tempoStore.tapTempo()
// - Save: call trackService.saveAnnotation()
```

#### components/Editing/RegionList.tsx
**Purpose**: List of regions with label dropdowns

**Implementation Guide**:
```typescript
import { useStructureStore } from "@/stores";
import { formatTime } from "@/utils/timeFormat";

// Display:
// - Each region: start-end time, duration, label dropdown
// - Click region to jump to it
// - Delete region button

// Label dropdown options:
// - intro, buildup, breakdown, breakdown-buildup, outro, default

// Connect to:
// - structureStore.regions (read)
// - structureStore.setRegionLabel() (write)
```

### Priority 4: Track Management

#### components/TrackList/TrackSelector.tsx
**Purpose**: Sidebar with track list

**Implementation Guide**:
```typescript
import { useTrackStore } from "@/stores";
import { trackService } from "@/services/api";
import { useWaveformStore, useStructureStore, useTempoStore } from "@/stores";

// Display:
// - Scrollable list of tracks
// - Indicator for reference/generated annotations
// - Selected state highlighting

// On track click:
// 1. Load track data: trackService.loadTrack(filename)
// 2. Update waveformStore with waveform data
// 3. Update tempoStore with BPM/downbeat
// 4. Load audio URL: trackService.getAudioUrl(filename)
// 5. Set audio element src
```

### Priority 5: Custom Hooks

#### hooks/useAudioPlayback.ts
**Purpose**: Manage audio element and playback state

**Implementation Guide**:
```typescript
// Create audio element on mount
// - Listen to play/pause/ended/timeupdate events
// - Update audioStore.isPlaying, audioStore.currentTime
// - Use requestAnimationFrame for smooth currentTime updates

// Return:
// - audioElement ref
// - Play/pause/seek functions
```

#### hooks/useWaveformInteraction.ts
**Purpose**: Handle mouse interaction with waveform

**Implementation Guide**:
```typescript
// Mouse events on waveform:
// - mousedown: start drag OR set cue point
// - mousemove: pan viewport if dragging
// - mouseup: end drag OR add boundary (Shift+Click)
// - wheel: zoom

// Distinguish click vs drag:
// - Track drag distance, threshold ~5px
// - If distance < threshold: click (set cue/add boundary)
// - If distance >= threshold: drag (pan)

// Apply quantization:
// - Check uiStore.quantizeEnabled
// - If enabled, use tempoStore.quantizeToBeat()
```

#### hooks/useKeyboardShortcuts.ts
**Purpose**: Handle keyboard shortcuts

**Implementation Guide**:
```typescript
// Shortcuts (matching original app):
// - Space: play/pause
// - B: add boundary at playhead
// - D: set downbeat at playhead
// - Q: toggle quantize
// - C/R: return to cue
// - Arrow Left/Right: jump (with Shift/Ctrl modifiers)
// - Arrow Up/Down: previous/next track
// - +/-: zoom
// - 0: zoom to fit

// Implementation:
// - useEffect with keydown listener
// - Check if input/select focused (ignore if so)
// - Dispatch to appropriate store actions
```

## Development Workflow

### Setup
```bash
cd packages/edm-annotator/frontend
pnpm install
```

### Run Development Server
```bash
# Terminal 1: Backend
cd packages/edm-annotator/backend
uv pip install -e .
edm-annotator --env development

# Terminal 2: Frontend
cd packages/edm-annotator/frontend
pnpm dev

# Open http://localhost:5174
```

### Build for Production
```bash
pnpm build
# Output: dist/
```

## Testing

### Setup Vitest
Create `tests/setup.ts`:
```typescript
import "@testing-library/jest-dom";
```

Update `vite.config.ts`:
```typescript
export default defineConfig({
  // ... existing config
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
  },
});
```

### Run Tests
```bash
pnpm test
pnpm test:coverage
```

### Example Test
```typescript
// tests/stores/audioStore.test.ts
import { renderHook, act } from "@testing-library/react";
import { useAudioStore } from "@/stores/audioStore";

describe("audioStore", () => {
  it("updates current time", () => {
    const { result } = renderHook(() => useAudioStore());
    act(() => result.current.updateCurrentTime(42.5));
    expect(result.current.currentTime).toBe(42.5);
  });
});
```

## Styling

The original app uses a dark theme with:
- Background: `#0F1419`
- Cards: `#1E2139`
- Primary: `#5B7CFF` (blue)
- Accent: `#00E6B8` (cyan)
- Warning: `#FFB800` (orange)
- Error: `#FF6B6B` (red)

Style components using CSS modules or styled-components as preferred.

## Architecture Decisions

### Why Zustand?
- Lightweight (2KB)
- TypeScript-first
- No boilerplate
- Direct store access (no Context Provider needed)

### Why SVG for Waveform?
- Crisp at any zoom level
- Easy hit detection
- Can overlay interactive elements
- Good performance with viewport culling

### Why Multiple Stores?
- Better performance (components only re-render when their store updates)
- Clearer separation of concerns
- Easier to test individual stores

## Implementation Timeline

Estimated effort for complete frontend:
- **Waveform visualization**: 4-6 hours
- **Transport & editing UI**: 2-3 hours
- **Track selector**: 1-2 hours
- **Hooks & interaction**: 2-3 hours
- **Polish & testing**: 2-3 hours

**Total: ~12-17 hours** for full feature parity with original app.

## Reference

Original app behaviors to replicate:
- Waveform: packages/edm-annotator/templates/index.html (lines 667-1737)
- Backend API: packages/edm-annotator/backend/src/edm_annotator/api/

For questions, see the implementation plan at `.claude/plans/velvety-tumbling-riddle.md`.

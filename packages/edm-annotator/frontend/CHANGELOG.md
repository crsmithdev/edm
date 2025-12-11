# Changelog

All notable changes to the EDM Annotator frontend.

## [Unreleased]

### Added

#### Dual Waveform Display System
- **OverviewWaveform**: Full track visualization with moving playhead
  - 60px compact view extending upward from baseline
  - Click-to-seek with quantize-aware bar snapping
  - Subtle region overlays (colored backgrounds, 0.15 opacity)
  - Purple boundary markers (2px width, 0.7 opacity)
  - Unlabeled regions show no overlay
- **DetailWaveform**: Centered playhead with scrolling waveform
  - 3-band mirrored display (bass cyan, mids purple, highs pink)
  - Fixed playhead at center, waveform scrolls underneath
  - Configurable time span (4-60s, default 16s)
  - Global amplitude scaling for consistent visual height
- **WaveformContainer**: Manages both views with zoom controls (×/÷ 1.5)

#### Drag-to-Scrub Playback Control
- Click and drag on detail waveform to scrub playback position
  - Drag left: Move forward in time
  - Drag right: Move backward in time
- Beat snapping during drag when quantize enabled
- **Shift+drag**: Bypasses quantize for fine scrubbing control
- Visual feedback with grab/grabbing cursors

#### Cue Point System
- **'C' key**: Sets cue point when stopped, returns to cue when playing
- **'R' key**: Always returns to cue point
- Visual cue point indicator
  - Orange gradient line (#ff9500 → #ff6b00)
  - Scrolls with waveform (unlike fixed playhead)
  - Snaps to nearest beat when quantize enabled
  - Only visible when within viewport
- Status messages for user feedback

#### Boundary Marking
- **Ctrl+click** (or Cmd+click on Mac) adds boundary at clicked position
- **'B' key**: Adds boundary at current playhead position
- Boundaries snap to nearest beat when quantize enabled
- Visual boundary markers
  - Purple color (#7b6aff) matching "+ Boundary" button
  - 3px width on detail waveform
  - 2px width on overview waveform
  - Hover tooltips showing time
  - Ctrl+click to delete
- Delete button in region list for removing boundaries

#### Beat Grid Visualization
- Adaptive grid density based on zoom level
- **Bar lines**: Grey (rgba(200,200,200,0.7)), 2px width
  - Bar numbers positioned 4px to right of line
  - Light grey text (rgba(220,220,220,0.9))
  - No background on labels
- **Beat lines**: Lighter grey (rgba(150,150,150,0.3)), 1px width
- **Downbeat**: Red (#f44336), 3px width
- Grid automatically hides when zoomed out to prevent clutter

#### Region Visualization
- Colored overlays for labeled sections (0.4 opacity on detail, 0.15 on overview)
- Section label colors:
  - intro: Purple (#7b6aff)
  - buildup: Orange (#ff9800)
  - main: Green (#4caf50)
  - breakdown: Blue (#2196f3)
  - outro: Purple (#9c27b0)
- Unlabeled regions show no overlay (filtered out)
- RegionList component with label dropdowns and delete buttons

#### Keyboard Shortcuts
- **Space**: Play/pause toggle
- **C**: Set cue point (stopped) / Return to cue (playing)
- **R**: Return to cue point
- **B**: Add boundary at playhead
- **D**: Set downbeat at playhead
- **Q**: Toggle quantize mode
- **←/→**: Navigate backward/forward by 1 bar
- **Shift+←/→**: Navigate by 4 bars
- **+/-**: Zoom in/out on detail waveform
- **Drag**: Scrub playback position
- **Shift+Drag**: Scrub (bypass quantize)
- **Ctrl+Click**: Add boundary at click position
- **?**: Show keyboard shortcuts popup

#### UI Components
- **PlaybackControls**: Play/pause button, previous/next track navigation
- **EditingControls**: Boundary, downbeat, and quantize toggle buttons
- **RegionList**: Table view of regions with inline label editing
- **TrackSelector**: Sidebar for track selection and loading
- **KeyboardHints**: Modal popup showing all shortcuts (triggered by '?' key or emoji link)
- **StatusMessages**: Toast-style notifications for user actions

#### Interaction Patterns
- **Cursors**:
  - Overview: Default arrow cursor
  - Detail: Grab cursor by default
  - Detail with Ctrl held: Crosshair (boundary mode)
  - Detail while dragging: Grabbing cursor
- **Quantize behavior**: Affects boundaries, cue points, and scrubbing
  - Shift key bypass for fine control when needed
- **Viewport synchronization**: Overlays receive actual viewport (including negative times at track start)

### Infrastructure

#### State Management
- **Zustand stores** replacing global variables:
  - `audioStore`: Playback state, cue points, seek operations
  - `waveformStore`: Waveform data, viewport, zoom level
  - `structureStore`: Boundaries and regions
  - `tempoStore`: BPM, downbeat, time/bar conversions
  - `uiStore`: Quantize, dragging state, status messages
  - `trackStore`: Track selection and loading

#### Type System
- Complete TypeScript interfaces for:
  - Track metadata
  - Waveform data (3-band frequency)
  - Structure annotations (boundaries, regions)
  - API requests/responses
- Strict mode enabled throughout
- No `any` types in codebase

#### Utilities
- Time formatting (seconds → MM:SS)
- Bar/beat calculations with BPM
- Time-to-bar and bar-to-time conversions
- Beat duration calculations
- Quantization helpers
- Color mappings for region labels

#### API Service
- Axios-based client with typed endpoints
- Endpoints for:
  - Track listing and loading
  - Waveform data fetching
  - Annotation loading/saving
  - Audio file serving

#### Build & Development
- Vite build system with Hot Module Replacement
- TypeScript 5.7+ strict mode
- ESLint configuration
- Prettier code formatting
- Development proxy for backend API

### Fixed
- Waveform scaling bug: Amplitude now calculated from entire track, not just visible viewport
- Viewport synchronization: Overlays now receive actual viewport including negative start times
- Side effect in useMemo: Changed to useEffect for viewport updates

### Performance
- Viewport culling: Only processes visible waveform samples
- Downsampling: Overview waveform uses max 500 samples
- Memoization: Heavy calculations wrapped in useMemo
- Callback optimization: Event handlers wrapped in useCallback
- Store granularity: Multiple focused stores prevent unnecessary re-renders

---

## Architecture

See `ARCHITECTURE.md` for detailed documentation of:
- Dual waveform system design
- Zustand store architecture and state flow
- Component hierarchy and responsibilities
- Interaction patterns and quantize behavior
- Visual design system (colors, indicators, overlays)

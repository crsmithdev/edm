# EDM Structure Annotator v2.0

Production-ready web application for annotating EDM track structure boundaries. Complete rewrite from monolithic prototype to modern architecture.

## Architecture Overview

```
packages/edm-annotator/
├── backend/                    # Flask API (COMPLETE ✅)
│   ├── src/edm_annotator/
│   │   ├── app.py             # Application factory
│   │   ├── config.py          # Environment configs
│   │   ├── api/               # Route blueprints
│   │   ├── services/          # Business logic layer
│   │   └── models/            # DTOs
│   └── tests/                 # Pytest tests
│
├── frontend/                   # React + TypeScript (FOUNDATION COMPLETE ✅)
│   ├── src/
│   │   ├── stores/            # Zustand state management (6 stores)
│   │   ├── types/             # TypeScript interfaces
│   │   ├── services/          # API client
│   │   ├── utils/             # Helper functions
│   │   └── components/        # React components (TO IMPLEMENT 🚧)
│   └── tests/
│
├── templates/                  # Legacy HTML (reference)
└── README.md                   # This file
```

## ✅ Completed Work

### Backend (100% Complete)
- **Application Factory Pattern**: Eliminates all global variables, enables testing
- **Service Layer**: Clean separation of concerns (Audio, Waveform, Annotation services)
- **API Blueprints**: Modular route organization (tracks, audio, waveforms, annotations)
- **Configuration Management**: Development/Production/Testing environments
- **Testing Infrastructure**: pytest with fixtures, conftest setup
- **Security**: Path traversal validation, CORS support
- **Dependencies**: pyproject.toml with all requirements

**Key Improvements Over Original**:
- ✅ No global variables (Flask app, paths all injected)
- ✅ Testable architecture (dependency injection)
- ✅ Production-ready configuration
- ✅ Proper error handling
- ✅ Type hints throughout

### Frontend (~90% Complete ✅)
- **Project Setup**: Vite + React 18 + TypeScript strict mode ✅
- **State Management**: 6 Zustand stores replacing 20+ global variables ✅
  - audioStore (playback state, cue points) ✅
  - trackStore (track selection) ✅
  - waveformStore (waveform data, zoom, viewport) ✅
  - structureStore (boundaries, regions) ✅
  - tempoStore (BPM, downbeat, calculations) ✅
  - uiStore (UI state, quantize, status) ✅
- **Type System**: Complete TypeScript interfaces ✅
- **Utilities**: Time formatting, bar calculations, quantization, colors ✅
- **API Service**: Axios-based client with typed endpoints ✅

**Implemented Components**:
- ✅ **Dual Waveform Display**
  - `WaveformContainer.tsx` - Manages overview + detail views
  - `OverviewWaveform.tsx` - Full track view with moving playhead
  - `DetailWaveform.tsx` - Centered playhead with 3-band waveform
- ✅ **Waveform Overlays**
  - `BeatGrid.tsx` - Adaptive bar/beat grid
  - `BoundaryMarkers.tsx` - Structure boundary markers
  - `RegionOverlays.tsx` - Colored section overlays
- ✅ **Controls**
  - `PlaybackControls.tsx` - Play/pause/track navigation
  - `EditingControls.tsx` - Boundary/downbeat/quantize controls
  - `RegionList.tsx` - Region editor with labels
  - `TrackSelector.tsx` - Track list sidebar
- ✅ **Hooks**
  - `useKeyboardShortcuts.ts` - Complete shortcut system
  - `useAudioPlayback.ts` - Audio element management

**Implemented Features**:
- ✅ Drag-to-scrub playback (waveform moves under fixed playhead)
- ✅ Cue point system (C/R keys, visual orange indicator)
- ✅ Quantize snapping (with Shift bypass for fine control)
- ✅ Boundary marking (Ctrl+click, snaps to beat)
- ✅ Click-to-seek in overview (snaps to bar when quantize on)
- ✅ Region labeling and visualization
- ✅ Complete keyboard shortcuts
- ✅ Track loading and annotation saving

**Architecture Decisions**:
- ✅ Zustand for lightweight, TypeScript-first state management
- ✅ Multiple focused stores (better performance, testability)
- ✅ Complete type safety (no `any` types)
- ✅ SVG waveforms with viewport culling

## 🚧 Remaining Work

### Frontend Polish & Testing (~10% remaining)
- [ ] Component tests (React Testing Library)
- [ ] Store tests (Zustand)
- [ ] E2E tests with real audio files
- [ ] Accessibility improvements (ARIA labels, keyboard nav)
- [ ] Error boundaries and loading states
- [ ] Performance optimization for very long tracks

### Backend Testing (Optional)
- [ ] Unit tests for services (audio, waveform, annotation)
- [ ] Integration tests for API endpoints

See `frontend/ARCHITECTURE.md` for detailed system documentation.

## Development Setup

### Prerequisites
- Python 3.12+
- Node.js 20+
- npm (comes with Node.js)
- uv (Python package manager)

### Initial Setup

**IMPORTANT**: Run setup commands from the workspace root (`/home/crsmi/edm`), not from inside package directories. The project uses a uv workspace that links dependencies between packages.

```bash
# From workspace root (/home/crsmi/edm)
uv sync  # Install all Python dependencies including edm-lib

# Install frontend dependencies
cd packages/edm-annotator/frontend
npm install
cd ../../..  # Back to workspace root
```

### Running the Dev Server

The easiest way to run both backend and frontend together:

```bash
# From packages/edm-annotator directory
./run-dev.sh
```

This starts both servers with proper logging and auto-reload.

### Running Servers Individually

#### Backend
```bash
# From workspace root
uv run edm-annotator --env development --port 5000

# Run tests (from workspace root)
uv run pytest packages/edm-annotator/backend/tests
```

#### Frontend
```bash
cd packages/edm-annotator/frontend
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

### Environment Variables
```bash
export EDM_AUDIO_DIR=/path/to/music        # Audio files directory
export EDM_ANNOTATION_DIR=/path/to/data    # Annotation output directory
```

## API Endpoints

### GET /api/tracks
List available audio files with annotation status
```json
[
  {
    "filename": "track.mp3",
    "path": "music/track.mp3",
    "has_reference": true,
    "has_generated": false
  }
]
```

### GET /api/load/<filename>
Load track waveform and metadata
```json
{
  "filename": "track.mp3",
  "duration": 240.5,
  "bpm": 128.0,
  "downbeat": 0.0,
  "sample_rate": 22050,
  "waveform_bass": [...],
  "waveform_mids": [...],
  "waveform_highs": [...],
  "waveform_times": [...]
}
```

### POST /api/save
Save annotation to YAML
```json
{
  "filename": "track.mp3",
  "bpm": 128.0,
  "downbeat": 0.0,
  "boundaries": [
    {"time": 0.0, "label": "intro"},
    {"time": 15.2, "label": "buildup"}
  ]
}
```

### GET /api/audio/<filename>
Serve audio file for playback (binary data)

## Architecture Highlights

### Backend: Application Factory Pattern
```python
def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class_map[config_name])

    # Dependency injection - no globals!
    audio_service = AudioService(config=app.config)
    waveform_service = WaveformService(config=app.config, audio_service=audio_service)

    app.audio_service = audio_service
    app.waveform_service = waveform_service

    register_routes(app)
    return app
```

### Frontend: Zustand State Management
```typescript
// Clean, typed stores replacing global variables
const useAudioStore = create<AudioState>((set, get) => ({
  currentTime: 0,
  isPlaying: false,
  play: () => { /* ... */ },
  seek: (time) => { /* ... */ },
}));

// Usage in components
function PlaybackControls() {
  const { play, pause, isPlaying } = useAudioStore();
  return <button onClick={isPlaying ? pause : play}>
    {isPlaying ? "Pause" : "Play"}
  </button>;
}
```

## Migration from v1.0

The original monolithic app (`templates/index.html`, `src/edm_annotator/app.py`) has been:
- **Backend**: Refactored into services, blueprints, proper config
- **Frontend**: Prepared for React migration with full state architecture

**Key Benefits**:
- 🚀 No global variables (testable, maintainable)
- 🔒 Secure (path validation, environment-based config)
- 📦 Modular (services, components)
- ✅ Tested (infrastructure ready)
- 🎨 Type-safe (TypeScript strict mode)

## Contributing

### Code Style
- **Backend**: Black formatter, Ruff linter, mypy type checking
- **Frontend**: ESLint, TypeScript strict mode

### Testing
- **Backend**: pytest with 85%+ coverage target
- **Frontend**: Vitest + React Testing Library, 70%+ coverage

### Workflow
1. Create feature branch (`feature/component-name`)
2. Implement with tests
3. Run quality checks (`pytest`, `pnpm test`, `pnpm lint`)
4. Submit PR with description

## References

- **Implementation Plan**: `.claude/plans/velvety-tumbling-riddle.md`
- **Frontend Guide**: `frontend/README.md`
- **Original Code**: `templates/index.html` (legacy reference)

## License

Part of the EDM monorepo.

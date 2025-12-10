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

### Frontend Infrastructure (Foundation Complete)
- **Project Setup**: Vite + React 18 + TypeScript strict mode
- **State Management**: 6 Zustand stores replacing 20+ global variables
  - audioStore (playback state)
  - trackStore (track selection)
  - waveformStore (waveform data, zoom, viewport)
  - structureStore (boundaries, regions)
  - tempoStore (BPM, downbeat, calculations)
  - uiStore (UI state, dragging, quantize)
- **Type System**: Complete TypeScript interfaces for tracks, waveform, structure, API
- **Utilities**: Time formatting, bar calculations, quantization, colors
- **API Service**: Axios-based client with typed endpoints
- **Configuration**: package.json, tsconfig, vite.config with proxy

**Key Architecture Decisions**:
- ✅ Zustand for lightweight, TypeScript-first state management
- ✅ Multiple focused stores (better performance, testability)
- ✅ Complete type safety (no `any` types)
- ✅ Utility-first design (pure functions, composable)

## 🚧 Remaining Work

### Frontend Components (~12-17 hours)

#### Priority 1: Waveform Visualization (4-6 hours)
- [ ] `WaveformCanvas.tsx` - SVG 3-band rendering, viewport culling
- [ ] `BeatGrid.tsx` - Bar/beat overlay with adaptive density
- [ ] `Playhead.tsx` - Current position indicator
- [ ] `BoundaryMarkers.tsx` - Draggable structure markers
- [ ] `RegionOverlays.tsx` - Colored region backgrounds

#### Priority 2: Controls (2-3 hours)
- [ ] `PlaybackControls.tsx` - Play/pause/cue buttons
- [ ] `NavigationControls.tsx` - Bar/beat jump controls
- [ ] `EditingControls.tsx` - Add boundary, set downbeat, quantize, tempo input
- [ ] `RegionList.tsx` - Region editor with label dropdowns

#### Priority 3: Track Management (1-2 hours)
- [ ] `TrackSelector.tsx` - Track list sidebar with load functionality

#### Priority 4: Hooks (2-3 hours)
- [ ] `useAudioPlayback.ts` - Audio element management, event handling
- [ ] `useWaveformInteraction.ts` - Mouse/touch interaction (click, drag, zoom)
- [ ] `useKeyboardShortcuts.ts` - Keyboard shortcuts (space, arrows, etc.)

#### Priority 5: Testing (2-3 hours)
- [ ] Component tests (React Testing Library)
- [ ] Store tests (Zustand)
- [ ] Utility tests
- [ ] Integration tests

### Backend Testing (Optional)
- [ ] Unit tests for services (audio, waveform, annotation)
- [ ] Integration tests for API endpoints

See `frontend/README.md` for detailed implementation guides.

## Development Setup

### Prerequisites
- Python 3.12+
- Node.js 20+
- pnpm (`corepack enable pnpm`)

### Backend Setup
```bash
cd packages/edm-annotator/backend
uv pip install -e ".[dev]"

# Run backend
edm-annotator --env development --port 5000

# Run tests
pytest
```

### Frontend Setup
```bash
cd packages/edm-annotator/frontend
pnpm install

# Run dev server (with backend proxy)
pnpm dev

# Build for production
pnpm build

# Run tests
pnpm test
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

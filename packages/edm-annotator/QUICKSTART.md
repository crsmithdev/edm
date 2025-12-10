# EDM Annotator - Quick Start Guide

## Prerequisites

- Python 3.12+
- Node.js 20+
- pnpm (run `corepack enable pnpm`)
- Audio files in a directory

## Setup (First Time)

### 1. Backend Setup

```bash
cd packages/edm-annotator/backend
uv pip install -e ".[dev]"
```

### 2. Frontend Setup

```bash
cd packages/edm-annotator/frontend
pnpm install
```

### 3. Environment Configuration

Set your audio directory (optional - defaults to ~/music):

```bash
export EDM_AUDIO_DIR=/path/to/your/music
export EDM_ANNOTATION_DIR=/path/to/annotations  # Optional - defaults to data/annotations
```

## Running the Application

### Terminal 1: Start Backend

```bash
cd packages/edm-annotator/backend
edm-annotator --env development --port 5000
```

You should see:
```
Environment: development
Audio directory: /path/to/music
Annotation directory: /path/to/annotations

Starting annotation server on http://0.0.0.0:5000
Debug mode enabled - auto-reload active
```

### Terminal 2: Start Frontend

```bash
cd packages/edm-annotator/frontend
pnpm dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### Open Application

Navigate to **http://localhost:5173** in your browser.

## Usage

### 1. Load a Track
- Click on a track in the right sidebar
- Wait for waveform to load
- Audio will be ready to play

### 2. Set Tempo
- Play the track and tap the "Tap" button on the beat
- Or manually enter BPM
- Click "Set Downbeat (D)" at the first beat of bar 1

### 3. Add Boundaries
- Click "Add Boundary (B)" button, OR
- Click on the waveform (Shift+Click), OR
- Press **B** key while at desired position

### 4. Label Regions
- Boundaries create regions automatically
- Use dropdowns in the Region List to set labels:
  - intro
  - buildup
  - breakdown
  - breakbuild
  - outro

### 5. Save Annotation
- Click "Save Annotation" button
- Annotations saved to `$EDM_ANNOTATION_DIR/reference/`

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Play/Pause |
| **B** | Add boundary at current position |
| **D** | Set downbeat at current position |
| **Q** | Toggle beat quantization |
| **C** or **R** | Return to cue point |
| **←** / **→** | Jump backward/forward (4 bars) |
| **Shift** + **←/→** | Jump 8 bars |
| **Ctrl/Cmd** + **←/→** | Jump 1 bar |
| **↑** / **↓** | Previous/Next track |
| **+** / **-** | Zoom in/out |
| **0** | Reset zoom |

## Tips

### Efficient Workflow
1. Load track
2. Tap tempo while listening
3. Set downbeat at bar 1
4. Use **Space** to play/pause
5. Press **B** when you hear structure changes
6. Use arrow keys to navigate
7. Label regions quickly with dropdowns
8. Save and move to next track (↓)

### Quantization
- Enable with **Q** key or toggle button
- Snaps clicks/boundaries to nearest beat
- Useful for precise alignment

### Zoom & Pan
- **Scroll wheel** on waveform to zoom centered on cursor
- **Click and drag** on waveform to pan
- **Shift + Click** adds boundary (won't pan)

### Boundary Management
- **Ctrl/Cmd + Click** on boundary marker to remove it
- Boundaries automatically create regions between them
- First and last boundaries define full structure

## Troubleshooting

### Backend won't start
- Check Python version: `python --version` (needs 3.12+)
- Verify installation: `uv pip list | grep edm-annotator`
- Check audio directory exists: `ls $EDM_AUDIO_DIR`

### Frontend won't start
- Check Node version: `node --version` (needs 20+)
- Install dependencies: `pnpm install`
- Clear cache: `rm -rf node_modules .vite && pnpm install`

### No tracks showing
- Verify `EDM_AUDIO_DIR` points to directory with audio files
- Supported formats: .mp3, .flac, .wav, .m4a
- Check backend logs for errors

### Waveform not loading
- Check browser console (F12) for errors
- Verify backend is running on port 5000
- Check CORS is enabled in backend

### Audio won't play
- Check audio file is valid (try in media player)
- Verify browser supports format
- Check browser console for errors
- Some browsers block autoplay - click Play button

## Development

### Backend Tests
```bash
cd packages/edm-annotator/backend
pytest
```

### Frontend Type Check
```bash
cd packages/edm-annotator/frontend
pnpm type-check
```

### Frontend Lint
```bash
pnpm lint
```

### Build for Production
```bash
cd packages/edm-annotator/frontend
pnpm build
# Output in dist/
```

## Next Steps

- See **README.md** for full documentation
- See **frontend/README.md** for component details
- Check `.claude/plans/velvety-tumbling-riddle.md` for architecture

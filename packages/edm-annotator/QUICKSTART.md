# EDM Annotator - Quick Start Guide

## Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- uv (Python package manager)
- Audio files in a directory

## Setup (First Time)

### 1. Install All Dependencies

From the repository root:

```bash
# Install backend (Python packages)
uv sync

# Install frontend (Node packages)
cd packages/edm-annotator/frontend
npm install
cd ../../..
```

### 2. Environment Configuration (Optional)

Set your audio directory (defaults to ~/music):

```bash
export EDM_AUDIO_DIR=/path/to/your/music
export EDM_ANNOTATION_DIR=/path/to/annotations  # Optional - defaults to data/annotations
```

## Running the Application

### Single Command (Recommended)

From the repository root:

```bash
just annotator
```

This starts both servers automatically:
- **Backend API**: http://localhost:5001
- **Frontend**: http://localhost:5174

You should see:
```
🎵 EDM Structure Annotator - Development Mode
==============================================

✅ Prerequisites met

Starting servers...
  Backend:  http://localhost:5001
  Frontend: http://localhost:5174

Press Ctrl+C to stop both servers
```

### Open Application

Navigate to **http://localhost:5174** in your browser.

---

## Manual Start (Alternative)

If you prefer to run servers separately:

### Terminal 1: Backend

```bash
cd packages/edm-annotator
uv run edm-annotator --env development --port 5001
```

### Terminal 2: Frontend

```bash
cd packages/edm-annotator/frontend
npm run dev
```

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
- Run from repo root: `uv sync` to install packages
- Verify: `uv run edm-annotator --help` should work
- Check audio directory exists: `ls $EDM_AUDIO_DIR`

### Frontend won't start
- Check Node version: `node --version` (needs 18+)
- Install dependencies: `cd packages/edm-annotator/frontend && npm install`
- Clear cache: `rm -rf node_modules .vite && npm install`

### No tracks showing
- Verify `EDM_AUDIO_DIR` points to directory with audio files
- Supported formats: .mp3, .flac, .wav, .m4a
- Check backend logs for errors

### Waveform not loading
- Check browser console (F12) for errors
- Verify backend is running on port 5001
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
npm run type-check
```

### Frontend Lint
```bash
cd packages/edm-annotator/frontend
npm run lint
```

### Build for Production
```bash
cd packages/edm-annotator/frontend
npm run build
# Output in dist/
```

## Next Steps

- See **README.md** for full documentation
- See **frontend/README.md** for component details
- Check `.claude/plans/velvety-tumbling-riddle.md` for architecture

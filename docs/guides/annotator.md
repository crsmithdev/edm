# Annotator Guide

Complete guide for the EDM Annotator web application - a tool for annotating track structure with boundaries and labels.

## Table of Contents

- [Quick Start](#quick-start)
- [Usage](#usage)
- [Development](#development)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+ (for frontend)
- uv (Python package manager)
- pnpm (Node package manager)
- Audio files in a directory

### Installation

From repository root:

```bash
# Install backend (Python packages)
uv sync

# Install frontend (Node packages)
cd packages/edm-annotator/frontend
pnpm install
cd ../../..
```

### Configuration

Set your audio directory (defaults to ~/music):

```bash
export EDM_AUDIO_DIR=/path/to/your/music
export EDM_ANNOTATION_DIR=/path/to/annotations  # Optional - defaults to data/annotations
```

### Running

**Single command** (recommended):

```bash
just annotator
```

This starts both servers:
- **Backend API**: http://localhost:5000
- **Frontend**: http://localhost:5174

Open **http://localhost:5174** in your browser.

---

## Usage

### Basic Workflow

1. **Load a Track**
   - Click on a track in the right sidebar
   - Wait for waveform to load
   - Audio will be ready to play

2. **Set Tempo**
   - Play the track and tap the "Tap" button on the beat
   - Or manually enter BPM
   - Click "Set Downbeat (D)" at the first beat of bar 1

3. **Add Boundaries**
   - Click "Add Boundary (B)" button, OR
   - Click on the waveform (Shift+Click), OR
   - Press **B** key while at desired position

4. **Label Regions**
   - Boundaries create regions automatically
   - Use dropdowns in the Region List to set labels:
     - intro
     - buildup
     - breakdown
     - drop
     - outro

5. **Save Annotation**
   - Click "Save Annotation" button
   - Annotations saved to `$EDM_ANNOTATION_DIR/reference/`

### Keyboard Shortcuts

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

### Tips

**Efficient workflow**:
1. Load track
2. Tap tempo while listening
3. Set downbeat at bar 1
4. Use **Space** to play/pause
5. Press **B** when you hear structure changes
6. Use arrow keys to navigate
7. Label regions quickly with dropdowns
8. Save and move to next track (↓)

**Quantization**:
- Enable with **Q** key or toggle button
- Snaps clicks/boundaries to nearest beat
- Useful for precise alignment

**Zoom & Pan**:
- **Scroll wheel** on waveform to zoom centered on cursor
- **Click and drag** on waveform to pan
- **Shift + Click** adds boundary (won't pan)

**Boundary Management**:
- **Ctrl/Cmd + Click** on boundary marker to remove it
- Boundaries automatically create regions between them
- First and last boundaries define full structure

---

## Development

### Architecture

```
┌─────────────┐     HTTP      ┌──────────────────┐
│   Browser   │ ←────────────→ │ Frontend (Vite)  │
└─────────────┘                │ Port: 5174       │
                               └──────────────────┘
                                        │
                                   WebSocket/HTTP
                                        ↓
                               ┌──────────────────┐
                               │ Backend (Flask)  │
                               │ Port: 5000       │
                               └──────────────────┘
                                        │
                                        ↓
                               ┌──────────────────┐
                               │  Audio Files     │
                               │  Annotations     │
                               └──────────────────┘
```

### Running Development Servers

#### Quick Start (Both Servers)

```bash
just annotator
```

#### Manual Start (Separate Terminals)

**Terminal 1 - Backend**:

```bash
cd packages/edm-annotator/backend
uv run edm-annotator --env development --port 5000
```

Backend serves:
- REST API endpoints at `/api/*`
- Health check at `/health`
- Audio file proxying

**Terminal 2 - Frontend**:

```bash
cd packages/edm-annotator/frontend
pnpm run dev
```

Frontend provides:
- Hot module replacement (HMR)
- React Fast Refresh
- TypeScript type checking
- Auto-reload on file changes

### Development Configuration

**Backend environment**:

```bash
export EDM_AUDIO_DIR=~/music              # Audio files directory
export EDM_ANNOTATION_DIR=data/annotations # Annotation save location
export EDM_LOG_LEVEL=DEBUG                 # Verbose logging
```

**Frontend build config**:

Vite config at `packages/edm-annotator/frontend/vite.config.ts`:
- Dev server port: 5174
- API proxy: http://localhost:5000
- HMR enabled by default

### Frontend Development

#### Type Checking

```bash
cd packages/edm-annotator/frontend
pnpm run type-check
```

#### Linting

```bash
pnpm run lint        # Check
pnpm run lint:fix    # Auto-fix
```

#### Testing

```bash
# Unit tests
pnpm test

# E2E tests (requires servers running)
pnpm run test:e2e
```

See `packages/edm-annotator/frontend/E2E_TESTING.md` for details.

#### Production Build

```bash
pnpm run build       # Output to dist/
pnpm run preview     # Preview production build
```

---

## Production Deployment

### Build Frontend

```bash
cd packages/edm-annotator/frontend
pnpm install
pnpm run build
```

Output: `dist/` directory with static files

### Configure Backend

```bash
cd packages/edm-annotator/backend

# Set environment variables
export EDM_AUDIO_DIR=/path/to/shared/audio
export EDM_ANNOTATION_DIR=/path/to/shared/annotations
export EDM_LOG_LEVEL=INFO
export FLASK_ENV=production
```

### Serve with Production WSGI Server

**Using Gunicorn** (recommended):

```bash
pip install gunicorn

gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 4 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  edm_annotator.app:app
```

### Serve Frontend with Nginx

```nginx
# /etc/nginx/sites-available/edm-annotator
server {
    listen 80;
    server_name annotator.example.com;

    # Frontend static files
    location / {
        root /path/to/edm/packages/edm-annotator/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://localhost:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Audio file serving
    location /audio/ {
        alias /path/to/shared/audio/;
        add_header Access-Control-Allow-Origin *;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/edm-annotator /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Systemd Service

Create `/etc/systemd/system/edm-annotator-backend.service`:

```ini
[Unit]
Description=EDM Annotator Backend
After=network.target

[Service]
Type=simple
User=edm
Group=edm
WorkingDirectory=/path/to/edm/packages/edm-annotator/backend
Environment="EDM_AUDIO_DIR=/path/to/audio"
Environment="EDM_ANNOTATION_DIR=/path/to/annotations"
Environment="EDM_LOG_LEVEL=INFO"
ExecStart=/usr/local/bin/gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 4 \
    --timeout 120 \
    edm_annotator.app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable edm-annotator-backend
sudo systemctl start edm-annotator-backend
sudo systemctl status edm-annotator-backend
```

### SSL/TLS (HTTPS)

Using Let's Encrypt:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d annotator.example.com
```

### Health Checks

```bash
# Check backend health
curl http://localhost:5000/health

# Expected response:
{"status": "ok"}
```

For complete deployment guide, see [Deployment Guide](../deployment.md).

---

## Troubleshooting

### Backend Issues

#### Backend won't start

**Solutions**:

1. **Check Python version**:
   ```bash
   python --version  # Needs 3.12+
   ```

2. **Install packages**:
   ```bash
   cd /path/to/edm
   uv sync
   ```

3. **Verify CLI works**:
   ```bash
   uv run edm-annotator --help
   ```

4. **Check audio directory**:
   ```bash
   ls $EDM_AUDIO_DIR  # Should list files
   ```

5. **Check port availability**:
   ```bash
   lsof -i :5000  # Should be empty
   # If occupied, kill process or use different port:
   uv run edm-annotator --port 5001
   ```

#### No tracks showing in sidebar

**Solutions**:

1. **Verify audio directory**:
   ```bash
   echo $EDM_AUDIO_DIR
   ls $EDM_AUDIO_DIR  # Should list audio files
   ```

2. **Check supported formats**: .mp3, .flac, .wav, .m4a

3. **Check backend logs**: Look for file scanning errors in terminal

4. **Verify backend is running**:
   ```bash
   curl http://localhost:5000/health
   # Should return: {"status": "ok"}
   ```

### Frontend Issues

#### Frontend won't start

**Solutions**:

1. **Check Node version**:
   ```bash
   node --version  # Needs 18+
   ```

2. **Install dependencies**:
   ```bash
   cd packages/edm-annotator/frontend
   pnpm install
   ```

3. **Clear cache**:
   ```bash
   cd packages/edm-annotator/frontend
   rm -rf node_modules .vite
   pnpm install
   ```

4. **Check port availability**:
   ```bash
   lsof -i :5174  # Should be empty
   ```

#### Waveform not loading

**Solutions**:

1. **Check browser console** (F12) for errors

2. **Verify backend is running**:
   ```bash
   curl http://localhost:5000/health
   ```

3. **Check CORS**: Backend should enable CORS in development mode

4. **Try different browser**: Some browsers have stricter security policies

5. **Check audio file size**: Very large files (>100MB) may be slow

#### Frontend not connecting to backend

**Solutions**:

1. **Verify backend is running**:
   ```bash
   curl http://localhost:5000/health
   ```

2. **Check browser console** (F12) for CORS errors

3. **Verify Vite proxy configuration** in `vite.config.ts`

#### Changes not reflecting

**Solutions**:

Frontend (should auto-reload via HMR):
```bash
# If stuck, clear cache:
rm -rf packages/edm-annotator/frontend/.vite
pnpm run dev
```

Backend (restart dev server):
```bash
# Or use auto-reload:
FLASK_DEBUG=1 uv run edm-annotator --env development
```

### Usage Issues

#### Audio won't play

**Solutions**:

1. **Check audio file validity**:
   ```bash
   ffplay audio/track.mp3  # Should play in terminal
   ```

2. **Check browser support**: Some browsers don't support certain codecs

3. **Check browser console** (F12) for errors

4. **Disable autoplay blocking**: Some browsers block autoplay - click Play button manually

5. **Try different format**: Convert to widely-supported format:
   ```bash
   ffmpeg -i track.flac track.mp3
   ```

#### Boundaries not snapping to beats

**Cause**: Quantization disabled or incorrect BPM/downbeat

**Solutions**:

1. **Enable quantization**: Press **Q** key or toggle button

2. **Re-tap tempo**: Delete BPM and tap again while listening

3. **Adjust downbeat**: Press **D** at actual first beat of bar 1

#### Save button disabled

**Cause**: Missing required data

**Checklist**:
- [ ] BPM is set (tap or manual)
- [ ] Downbeat is set (press D)
- [ ] At least one boundary exists
- [ ] All regions have labels

**Solution**: Complete missing items and save button will enable

### Port Conflicts

**Port already in use**:

```bash
# Find process using port
lsof -i :5000
lsof -i :5174

# Kill if needed or use different port
uv run edm-annotator --port 5001
```

### Performance Issues

#### High memory usage

**Solutions**:

- Check for memory leaks
- Reduce Gunicorn workers
- Monitor with: `htop` or `free -h`
- Check for large files in temp directories

---

## See Also

- **[Deployment Guide](../deployment.md)** - Complete production deployment
- **[Troubleshooting Guide](../reference/troubleshooting.md)** - General troubleshooting
- **[Development Setup](../development/setup.md)** - Development environment
- **Package Documentation**:
  - [edm-annotator README](../../packages/edm-annotator/README.md)
  - [Frontend README](../../packages/edm-annotator/frontend/README.md)
  - [Architecture Details](../../packages/edm-annotator/ARCHITECTURE.md)

# Enhanced Dev Server Logging - Preview

## New Features

### 1. Rich Startup Banner

The backend now displays a colorful, organized startup banner with:
- Configuration details (environment, debug mode, auto-reload status)
- Directory paths (audio, annotations)
- Data statistics (audio file count, annotation count, supported formats)
- Server information (URL, CORS settings)
- Status indicators and warnings

### 2. Improved Shell Script Output

The `run-dev.sh` script now shows:
- Styled header with box drawing characters
- Color-coded prerequisite checks
- Server information table
- Process IDs for tracking
- Graceful shutdown messages

### 3. Request Logging (Development Mode)

API requests are now logged with:
- Color-coded HTTP methods (GET=cyan, POST=green, etc.)
- Request paths
- Color-coded status codes (2xx=green, 4xx=yellow, 5xx=red)
- Response time with performance-based coloring (<50ms=green, <200ms=yellow, >200ms=red)

## Example Output

### Startup (run-dev.sh)

```
╔══════════════════════════════════════════════════════════╗
║  🎵  EDM Structure Annotator - Development Mode      ║
╚══════════════════════════════════════════════════════════╝

Checking Prerequisites
  ✓ Backend installed
  ✓ Frontend dependencies installed

Starting Servers
  ┌────────────────────────────────────────────────────────┐
  │ Backend (Flask)    http://localhost:5000               │
  │ Frontend (Vite)    http://localhost:5174               │
  └────────────────────────────────────────────────────────┘

  Starting backend...
```

### Backend Startup

```
╔══════════════════════════════════════════════════════════╗
║  🎵  EDM Structure Annotator - Backend API          ║
╚══════════════════════════════════════════════════════════╝

Configuration
  ┌────────────────────────────────────────────────────────┐
  │ Environment     development                            │
  │ Debug Mode      enabled                                │
  │ Auto-reload     active                                 │
  └────────────────────────────────────────────────────────┘

Paths
  ┌────────────────────────────────────────────────────────┐
  │ Audio           /home/user/music                       │
  │ Annotations     /home/user/edm/data/annotations        │
  └────────────────────────────────────────────────────────┘

Available Data
  ┌────────────────────────────────────────────────────────┐
  │ Audio Files     42                                     │
  │ Annotations     15                                     │
  │ Formats         .mp3, .flac, .wav, .m4a               │
  └────────────────────────────────────────────────────────┘

Server
  ┌────────────────────────────────────────────────────────┐
  │ URL             http://0.0.0.0:5000                   │
  │ CORS            http://localhost:5174                  │
  └────────────────────────────────────────────────────────┘

  ✓  Ready to annotate 42 tracks

  Press Ctrl+C to stop the server
```

### Request Logging

```
  [api]      GET     /api/tracks                           [200]   12.3ms
  [api]      GET     /api/tracks/track1.mp3/waveform       [200]  145.7ms
  [api]      POST    /api/annotations                      [201]   23.4ms
  [api]      GET     /api/tracks/track2.mp3/audio          [200]   89.1ms
  [api]      DELETE  /api/annotations/track1.yaml          [204]    8.2ms
```

### Shutdown

```
Shutting Down
  Stopping backend (PID: 12345)
  Stopping frontend (PID: 12346)
  ✓ Servers stopped
```

## Color Scheme

- **Headers**: Cyan/Blue with bold
- **Success indicators**: Green (✓)
- **Warning indicators**: Yellow (⚠)
- **Error indicators**: Red (✗)
- **Paths/URLs**: Cyan
- **Metadata**: Dim gray
- **HTTP Methods**:
  - GET: Cyan
  - POST: Green
  - PUT/PATCH: Yellow
  - DELETE: Red
  - OPTIONS: Dim gray
- **Status Codes**:
  - 2xx: Green
  - 3xx: Cyan
  - 4xx: Yellow
  - 5xx: Red
- **Response Times**:
  - < 50ms: Green
  - < 200ms: Yellow
  - > 200ms: Red

## Usage

Just run as before:

```bash
just annotator
```

Or manually:

```bash
cd packages/edm-annotator
./run-dev.sh
```

The enhanced logging will automatically appear in development mode!

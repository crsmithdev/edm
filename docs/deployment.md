# Deployment Guide

Guide for deploying EDM components in production environments.

## Table of Contents

- [Current Status](#current-status)
- [Annotator Web Application](#annotator-web-application)
- [Model Inference (Planned)](#model-inference-planned)
- [Environment Configuration](#environment-configuration)
- [Production Checklist](#production-checklist)
- [Monitoring and Logging](#monitoring-and-logging)
- [Scaling Considerations](#scaling-considerations)

---

## Current Status

**Production-Ready Components**:
- ✅ edm-lib (Python library)
- ✅ edm-cli (command-line tool)
- ✅ edm-annotator (web application for local/team use)

**In Development**:
- 🚧 FastAPI inference service (see `openspec/changes/SERVE-fastapi-inference-service/`)
- 🚧 Docker containerization
- 🚧 Prometheus/Grafana monitoring (see `openspec/changes/MONITOR-prometheus-grafana-monitoring/`)

**Not Yet Implemented**:
- ❌ Kubernetes deployment
- ❌ Cloud deployment templates (AWS/GCP/Azure)
- ❌ Load balancing and auto-scaling
- ❌ Distributed training

---

## Annotator Web Application

The EDM Annotator can be deployed for team use on a shared server.

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

### Production Build

#### 1. Build Frontend

```bash
cd packages/edm-annotator/frontend
pnpm install
pnpm run build
```

Output: `dist/` directory with static files

#### 2. Configure Backend

```bash
cd packages/edm-annotator/backend

# Set environment variables
export EDM_AUDIO_DIR=/path/to/shared/audio
export EDM_ANNOTATION_DIR=/path/to/shared/annotations
export EDM_LOG_LEVEL=INFO
export FLASK_ENV=production
```

#### 3. Serve with Production WSGI Server

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

**Using uWSGI**:

```bash
pip install uwsgi

uwsgi \
  --http 0.0.0.0:5000 \
  --module edm_annotator.app:app \
  --processes 4 \
  --threads 2 \
  --enable-threads
```

#### 4. Serve Frontend

**Option 1: Nginx (recommended)**

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

**Option 2: Simple static server (development only)**

```bash
cd packages/edm-annotator/frontend/dist
python -m http.server 8080
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

**Using Let's Encrypt**:

```bash
sudo apt install certbot python3-certbot-nginx

sudo certbot --nginx -d annotator.example.com

# Auto-renewal is configured by default
```

### Access Control

**Option 1: Basic HTTP Authentication (Nginx)**

```nginx
server {
    # ... other config ...

    location / {
        auth_basic "EDM Annotator";
        auth_basic_user_file /etc/nginx/.htpasswd;
        # ... rest of location config ...
    }
}
```

Create password file:

```bash
sudo apt install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd username
```

**Option 2: VPN or IP Whitelist**

```nginx
# Allow specific IPs only
allow 192.168.1.0/24;
allow 10.0.0.0/8;
deny all;
```

### Data Backup

```bash
# Backup annotations (important!)
rsync -av /path/to/annotations/ /backup/location/

# Or use cron
0 2 * * * rsync -av /path/to/annotations/ /backup/annotations-$(date +\%Y\%m\%d)/
```

### Health Checks

```bash
# Check backend health
curl http://localhost:5000/health

# Expected response:
{"status": "ok"}
```

---

## Model Inference (Planned)

**Status**: Not yet implemented. See `openspec/changes/SERVE-fastapi-inference-service/proposal.md`

### Planned Architecture

```
┌─────────┐     HTTP       ┌──────────────────┐
│ Client  │ ──────────────→ │  FastAPI Service │
└─────────┘                 │  Port: 8000      │
                            └──────────────────┘
                                     │
                                     ↓
                            ┌──────────────────┐
                            │ MLflow Registry  │
                            │ (Load Models)    │
                            └──────────────────┘
                                     │
                                     ↓
                            ┌──────────────────┐
                            │  Inference       │
                            │  (PyTorch)       │
                            └──────────────────┘
```

### Planned Endpoints

- `POST /predict` - Analyze audio file
- `GET /health` - Health check
- `GET /model/info` - Model metadata

### Planned Deployment

**Docker**:

```dockerfile
# Planned Dockerfile structure
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "edm.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose**:

```yaml
# Planned docker-compose.yml
version: '3.8'
services:
  edm-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=file:///mlruns
    volumes:
      - ./mlruns:/mlruns
      - ./models:/models
```

**Status**: Implementation tracked in `openspec/changes/SERVE-fastapi-inference-service/tasks.md`

---

## Environment Configuration

### Production Environment Variables

**Backend (Annotator)**:

```bash
# Required
export EDM_AUDIO_DIR=/path/to/audio
export EDM_ANNOTATION_DIR=/path/to/annotations

# Optional
export EDM_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
export EDM_LOG_FILE=/var/log/edm-annotator.log
export FLASK_ENV=production
export WORKERS=4                   # Gunicorn workers
```

**Frontend Build**:

```bash
# Build-time variables (in .env.production)
VITE_API_BASE_URL=https://api.example.com
VITE_WS_URL=wss://api.example.com
```

### Security Considerations

**Never expose**:
- MLflow tracking URI with write access
- Internal file paths
- Database credentials (if added)
- API keys (if added)

**Best practices**:
- Use environment variables, not hardcoded values
- Rotate credentials regularly
- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Keep dependencies updated

---

## Production Checklist

### Pre-deployment

- [ ] Run all tests: `pytest`
- [ ] Check code quality: `ruff check`
- [ ] Build frontend: `pnpm run build`
- [ ] Test production build locally
- [ ] Backup existing data
- [ ] Document configuration
- [ ] Set up monitoring (see below)

### Deployment

- [ ] Update dependencies: `uv sync`
- [ ] Set environment variables
- [ ] Configure web server (Nginx/Apache)
- [ ] Set up systemd service
- [ ] Configure SSL/TLS
- [ ] Set up log rotation
- [ ] Configure backups
- [ ] Test health checks
- [ ] Verify audio file access
- [ ] Test annotation saving

### Post-deployment

- [ ] Monitor logs for errors
- [ ] Check disk space usage
- [ ] Verify backups are working
- [ ] Test from client browsers
- [ ] Document deployment for team
- [ ] Set up alerts (disk space, errors)

---

## Monitoring and Logging

### Application Logs

**Backend logs**:

```bash
# Systemd service logs
sudo journalctl -u edm-annotator-backend -f

# Application logs (if EDM_LOG_FILE set)
tail -f /var/log/edm-annotator.log
```

**Nginx logs**:

```bash
# Access logs
tail -f /var/log/nginx/access.log

# Error logs
tail -f /var/log/nginx/error.log
```

### Log Rotation

Create `/etc/logrotate.d/edm-annotator`:

```
/var/log/edm-annotator.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 edm edm
    sharedscripts
    postrotate
        systemctl reload edm-annotator-backend
    endscript
}
```

### Metrics to Monitor

**System metrics**:
- CPU usage
- Memory usage
- Disk space (especially for audio/annotations)
- Network I/O

**Application metrics**:
- Request count
- Response times
- Error rates
- Active users (WebSocket connections)

**Business metrics**:
- Annotations created per day
- Tracks annotated
- User activity

### Planned Monitoring (Not Yet Implemented)

See `openspec/changes/MONITOR-prometheus-grafana-monitoring/proposal.md`:

- Prometheus metrics collection
- Grafana dashboards
- Alerting rules
- Custom metrics

---

## Scaling Considerations

### Annotator Scaling

**Vertical Scaling** (current approach):
- Increase CPU/RAM on server
- Use faster storage (SSD)
- Increase Gunicorn workers

**Horizontal Scaling** (requires work):
- Load balancer (Nginx, HAProxy)
- Multiple backend instances
- Shared storage for audio/annotations (NFS, S3)
- Session management (Redis)

**Bottlenecks to address**:
- Audio file I/O (use CDN or object storage)
- Waveform generation (cache or pre-generate)
- Concurrent write to annotation files (use database)

### Model Inference Scaling (Planned)

**Planned features**:
- GPU support for faster inference
- Request batching
- Model caching in memory
- Horizontal scaling with multiple replicas
- Async processing for large files

**Resource requirements** (estimated):
- CPU-only: 2-4 cores, 4GB RAM per instance
- GPU: 1x T4 or better, 8GB RAM
- Storage: ~1GB for model weights

---

## Database (Not Yet Used)

Current storage:
- Annotations: YAML files
- MLflow: SQLite (local)
- No persistent user data

**Future considerations**:
- PostgreSQL for annotations (multi-user)
- Redis for caching
- S3/MinIO for audio storage

---

## Cloud Deployment (Not Yet Documented)

**Planned platforms**:
- AWS: EC2, S3, RDS, ECS/EKS
- GCP: Compute Engine, Cloud Storage, Cloud SQL, GKE
- Azure: VMs, Blob Storage, PostgreSQL, AKS

**Not yet documented**:
- Cloud-specific deployment guides
- Terraform/CloudFormation templates
- CI/CD pipelines for cloud
- Cost optimization strategies

---

## Disaster Recovery

### Backup Strategy

**Critical data**:
- Annotations (most important!)
- Trained models
- MLflow metadata

**Backup frequency**:
- Annotations: Daily + after each session
- Models: After training runs
- MLflow: Weekly

**Backup locations**:
- On-site: `/backup/edm/`
- Off-site: Cloud storage (S3, Google Drive, etc.)

### Recovery Procedures

**Annotation loss**:

```bash
# Restore from backup
rsync -av /backup/annotations/ /path/to/annotations/

# Verify
ls -la /path/to/annotations/reference/
```

**Model loss**:

```bash
# Pull from DVC remote
dvc pull outputs/training/run_name.dvc

# Or restore from MLflow
mlflow artifacts download --run-id <run-id> --dst-path outputs/
```

**Complete system failure**:

1. Restore application code from git
2. Restore annotations from backup
3. Restore models from DVC/MLflow
4. Reconfigure environment variables
5. Test before allowing user access

---

## Security Best Practices

### Application Security

- Keep dependencies updated: `uv sync`
- Use HTTPS (Let's Encrypt)
- Implement authentication (basic auth, OAuth)
- Validate file uploads (if added)
- Sanitize user inputs
- Use CSP headers
- Enable CORS only for trusted origins

### Server Security

- Keep OS updated: `sudo apt update && sudo apt upgrade`
- Use firewall: `ufw enable`
- Disable unnecessary services
- Use SSH keys, disable password auth
- Regular security audits
- Monitor logs for suspicious activity

### Data Security

- Encrypt data at rest (if sensitive)
- Encrypt data in transit (HTTPS/WSS)
- Regular backups (3-2-1 rule)
- Access control on annotations
- Audit logs for data access

---

## Performance Optimization

### Backend

- Use Gunicorn with multiple workers
- Enable gzip compression in Nginx
- Cache waveform data
- Use CDN for static assets
- Optimize audio file serving
- Profile slow endpoints

### Frontend

- Build with production mode: `pnpm run build`
- Enable Vite minification
- Lazy load components
- Optimize images
- Use browser caching
- Monitor bundle size

### Database (when added)

- Index frequently queried fields
- Use connection pooling
- Cache query results
- Regular VACUUM (PostgreSQL)
- Monitor slow queries

---

## Troubleshooting Deployment

### Service won't start

```bash
# Check service status
sudo systemctl status edm-annotator-backend

# Check logs
sudo journalctl -u edm-annotator-backend -n 50

# Common issues:
# - Port already in use: check with `lsof -i :5000`
# - Environment variables not set
# - File permissions issues
# - Python environment issues
```

### Frontend shows blank page

```bash
# Check browser console (F12)
# Common issues:
# - API URL misconfigured
# - CORS errors (backend not allowing origin)
# - Build errors (check build output)
# - Assets not found (check Nginx config)
```

### Audio files not loading

```bash
# Check Nginx audio alias
# Check file permissions: `ls -la /path/to/audio`
# Check browser network tab (F12)
# Verify CORS headers
```

### High memory usage

```bash
# Check for memory leaks
# Reduce Gunicorn workers
# Monitor with: `htop` or `free -h`
# Check for large files in temp directories
```

---

## Additional Resources

- **Annotator Architecture**: `packages/edm-annotator/ARCHITECTURE.md`
- **Model Management Guide**: `docs/guides/model-management.md`
- **Development Setup**: `docs/development.md`
- **Troubleshooting**: `docs/troubleshooting.md`

---

## Future Work

Planned but not yet implemented:

1. **Docker deployment** - See `openspec/changes/SERVE-fastapi-inference-service/`
2. **FastAPI inference service** - REST API for model predictions
3. **Monitoring stack** - See `openspec/changes/MONITOR-prometheus-grafana-monitoring/`
4. **Kubernetes deployment** - Not yet planned
5. **Cloud templates** - AWS/GCP/Azure deployment guides
6. **CI/CD pipelines** - Automated deployment

Track progress in `openspec/changes/` directory.

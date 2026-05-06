# Texture Roughness Demo Deployment

This folder is self-contained for a Docker Compose deployment.

## Run

```bash
docker compose up --build
```

Open:

```text
http://SERVER_IP:8080
```

Local test URL:

```text
http://localhost:8080
```

## What Is Included

- `frontend/`: Three.js viewer built by Docker and served through Nginx
- `backend/`: FastAPI prediction API
- `texture-model-pipeline/texture_model.py`: PyTorch model architecture
- `outputs/unet_skip_connections.pth`: trained model weights

The frontend calls `/api/predict`, and Nginx proxies `/api` to the backend container.

## Change Port

Edit `docker-compose.yml`:

```yaml
ports:
  - "8080:80"
```

For example, use `"80:80"` if your server allows port 80.

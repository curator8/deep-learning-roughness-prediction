from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE_DIR = PROJECT_ROOT / "texture-model-pipeline"
if str(MODEL_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_SOURCE_DIR))

from texture_model import RoughnessUNet, predict_seamless_roughness  # noqa: E402


IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
CHECKPOINT_PATH = Path(
    os.getenv("MODEL_CHECKPOINT", PROJECT_ROOT / "outputs" / "unet_skip_connections.pth")
)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: RoughnessUNet | None = None

app = FastAPI(title="Roughness Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def load_model() -> RoughnessUNet:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found at {CHECKPOINT_PATH}")

    loaded_model = RoughnessUNet().to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()
    return loaded_model


@app.on_event("startup")
def startup() -> None:
    global model
    model = load_model()


def image_to_tensor(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Upload must be a valid image.") from exc

    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def roughness_to_png(roughness: torch.Tensor) -> bytes:
    image_array = roughness.squeeze(0).squeeze(0).cpu().clamp(0.0, 1.0).numpy()
    image_uint16 = (image_array * 65535.0).round().astype(np.uint16)

    buffer = io.BytesIO()
    Image.fromarray(image_uint16, mode="I;16").save(buffer, format="PNG")
    return buffer.getvalue()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "device": device.type,
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.post("/predict", response_class=Response)
async def predict(file: UploadFile = File(...)) -> Response:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Upload cannot be empty.")

    input_tensor = image_to_tensor(image_bytes)
    with torch.inference_mode():
        predicted_roughness = predict_seamless_roughness(model, input_tensor)

    return Response(
        content=roughness_to_png(predicted_roughness),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


"""VeriLens Forensics API and web interface."""

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from core.pipeline import VeriLensPipeline


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}
INDEX_FILE = Path(__file__).resolve().parents[1] / "public" / "index.html"

app = FastAPI(
    title="VeriLens Forensics API",
    description="Heuristic image-forensics indicators, not proof of authenticity.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

pipeline = VeriLensPipeline()


def array_to_base64_png(image: np.ndarray) -> str:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid visualization.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        encoded_input = image
    elif image.ndim == 3 and image.shape[2] == 3:
        encoded_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported visualization shape.")

    ok, buffer = cv2.imencode(".png", encoded_input)
    if not ok:
        raise ValueError("Could not encode visualization.")

    data = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


@app.get("/")
def homepage():
    if not INDEX_FILE.is_file():
        raise HTTPException(status_code=500, detail="Website file is missing.")
    return FileResponse(INDEX_FILE, media_type="text/html")


@app.get("/health")
@app.get("/api")
def health_check():
    return {
        "status": "online",
        "service": "VeriLens Forensics API",
        "version": "1.1.0",
        "analysis_type": "heuristic",
    }


@app.post("/scan")
@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):
    try:
        if not (file.content_type or "").lower().startswith("image/"):
            raise HTTPException(status_code=415, detail="Upload an image file.")

        content = await file.read(MAX_UPLOAD_BYTES + 1)

        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")

        try:
            with Image.open(io.BytesIO(content)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=415, detail="Unsupported image format.")
                probe.verify()

            with Image.open(io.BytesIO(content)) as source:
                image = source.convert("RGB")

        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid image file.")

        report = pipeline.analyze(image)
        visualizations = report.get("visualizations", {})

        fft = visualizations.get("fft_spectrum_heatmap")
        ela = visualizations.get("tampering_mask")

        return JSONResponse(content={
            "status": "success",
            "verdict": report.get("verdict", "INCONCLUSIVE"),
            "confidence": report.get("confidence"),
            "fake_probability": report.get("fake_probability"),
            "diagnostics": report.get("diagnostics", {}),
            "visualizations": {
                "fft_heatmap_url": array_to_base64_png(fft) if fft is not None else None,
                "tampering_mask_url": array_to_base64_png(ela) if ela is not None else None,
            },
            "notice": "These heuristic findings are not proof of authenticity or manipulation.",
        })

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check the server logs.",
        )
    finally:
        await file.close()

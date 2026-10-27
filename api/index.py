
"""VeriLens Forensics API.

Accepts image uploads and returns forensic indicators.
The analysis is heuristic and must not be treated as proof of authenticity.
"""

import base64
import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from core.pipeline import VeriLensPipeline


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

app = FastAPI(
    title="VeriLens Forensics API",
    description=(
        "Image forensic indicators for research and inspection. "
        "Results are heuristic and are not proof of authenticity."
    ),
    version="1.1.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("VERILENS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

pipeline = VeriLensPipeline()


def array_to_base64_png(image_array: np.ndarray) -> str:
    """Encode an RGB or grayscale NumPy image as a PNG data URI."""
    if not isinstance(image_array, np.ndarray) or image_array.size == 0:
        raise ValueError("Invalid visualization array.")

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 2:
        image_to_encode = image_array
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        image_to_encode = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported visualization dimensions.")

    success, encoded = cv2.imencode(".png", image_to_encode)
    if not success:
        raise ValueError("Visualization encoding failed.")

    encoded_text = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_text}"


@app.get("/")
@app.get("/api")
def health_check():
    return {
        "status": "online",
        "service": "VeriLens Forensics API",
        "version": "1.1.0",
        "analysis_type": "heuristic_forensic_indicators",
        "authenticity_guaranteed": False,
    }


@app.post("/scan")
@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Upload a supported image file.",
        )

    try:
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)

        if not image_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds the 10 MB upload limit.",
            )

        try:
            with Image.open(io.BytesIO(image_bytes)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="This image format is not supported.",
                    )
                probe.verify()

            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")

        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            )

        report = pipeline.analyze(image)
        visualizations = report.get("visualizations", {})

        fft_heatmap = visualizations.get("fft_spectrum_heatmap")
        tampering_mask = visualizations.get("tampering_mask")

        response_data = {
            "status": "success",
            "verdict": report.get("verdict", "INCONCLUSIVE"),
            "confidence": report.get("confidence"),
            "fake_probability": report.get("fake_probability"),
            "diagnostics": report.get("diagnostics", {}),
            "visualizations": {
                "fft_heatmap_url": (
                    array_to_base64_png(fft_heatmap)
                    if fft_heatmap is not None
                    else None
                ),
                "tampering_mask_url": (
                    array_to_base64_png(tampering_mask)
                    if tampering_mask is not None
                    else None
                ),
            },
            "notice": (
                "Results are heuristic indicators, not proof that an image "
                "is authentic, manipulated, or AI-generated."
            ),
        }

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Image analysis failed. Check server logs for details.",
        )
    finally:
        await file.close()

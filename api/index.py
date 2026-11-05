
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
"""VeriLens Forensics API and web interface."""

import base64
import io
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from core.pipeline import VeriLensPipeline


logger = logging.getLogger("verilens.api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

ALLOWED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
    "TIFF",
}

INDEX_FILE = (
    Path(__file__).resolve().parents[1]
    / "public"
    / "index.html"
)

app = FastAPI(
    title="VeriLens Forensics API",
    description=(
        "Multi-engine image forensic analysis. "
        "Findings are diagnostic indicators, not proof of authenticity."
    ),
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

pipeline = VeriLensPipeline()


def make_json_safe(value: Any) -> Any:
    """Recursively convert scientific Python objects into JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"f", "c"}:
            safe_array = np.where(
                np.isfinite(value),
                value,
                0,
            )
            return safe_array.tolist()

        return value.tolist()

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Image.Image):
        return None

    return str(value)


def array_to_base64_png(image: np.ndarray) -> str:
    """Encode a grayscale or RGB visualization as a PNG data URL."""

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid visualization.")

    image = np.asarray(image)

    if image.dtype != np.uint8:
        if image.dtype.kind == "f":
            image = np.nan_to_num(
                image,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        encoded_input = image

    elif image.ndim == 3 and image.shape[2] == 3:
        encoded_input = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

    elif image.ndim == 3 and image.shape[2] == 4:
        encoded_input = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2BGRA,
        )

    else:
        raise ValueError("Unsupported visualization shape.")

    success, buffer = cv2.imencode(".png", encoded_input)

    if not success:
        raise ValueError("Could not encode visualization.")

    encoded = base64.b64encode(
        buffer.tobytes()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def validate_uploaded_image(content: bytes) -> Image.Image:
    """Validate image format, integrity, dimensions, and pixel count."""

    try:
        with Image.open(io.BytesIO(content)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported image format.",
                )

            width, height = probe.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image dimensions.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image exceeds the maximum pixel limit.",
                )

            probe.verify()

        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        return image

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
        )


@app.get("/")
def homepage():
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=500,
            detail="Website file is missing.",
        )

    return FileResponse(
        INDEX_FILE,
        media_type="text/html",
    )


@app.get("/health")
@app.get("/api")
def health_check():
    return {
        "status": "online",
        "service": "VeriLens Forensics API",
        "version": "1.2.0",
        "analysis_type": "multi_engine_forensics",
        "engines": {
            "metadata": "available",
            "frequency": "available",
            "spatial_forensics": "available",
            "biometrics": "optional",
            "deepfake": "optional_model_required",
        },
    }


@app.post("/scan")
@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):

    try:
        content_type = (file.content_type or "").lower()

        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail="Upload an image file.",
            )

        content = await file.read(MAX_UPLOAD_BYTES + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Maximum image size is 10 MB.",
            )

        image = validate_uploaded_image(content)

        # Execute the unified forensic pipeline.
        report = pipeline.analyze(image)

        if not isinstance(report, dict):
            raise TypeError("Pipeline returned an invalid report.")

        visualizations = report.get("visualizations", {})

        fft = visualizations.get("fft_spectrum_heatmap")
        ela = visualizations.get("tampering_mask")

        try:
            fft_url = (
                array_to_base64_png(fft)
                if isinstance(fft, np.ndarray)
                else None
            )
        except (ValueError, cv2.error):
            logger.warning("FFT visualization encoding failed.")
            fft_url = None

        try:
            ela_url = (
                array_to_base64_png(ela)
                if isinstance(ela, np.ndarray)
                else None
            )
        except (ValueError, cv2.error):
            logger.warning("ELA visualization encoding failed.")
            ela_url = None

        diagnostics = make_json_safe(
            report.get("diagnostics", {})
        )

        response = {
            "status": "success",
            "verdict": report.get(
                "verdict",
                "INCONCLUSIVE",
            ),
            "verdict_basis": report.get("verdict_basis"),
            "confidence": report.get("confidence"),
            "fake_probability": report.get("fake_probability"),
            "risk_index": report.get("risk_index"),
            "score_type": report.get("score_type"),
            "limitations": report.get("limitations", []),
            "diagnostics": diagnostics,
            "visualizations": {
                "fft_heatmap_url": fft_url,
                "tampering_mask_url": ela_url,
            },
            "notice": (
                "Forensic findings are diagnostic indicators. "
                "They are not proof of authenticity or manipulation."
            ),
        }

        return JSONResponse(
            content=make_json_safe(response),
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Image analysis failed.")

        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Check the server logs.",
        )

    finally:
        await file.close()

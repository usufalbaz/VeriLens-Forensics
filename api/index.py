"""
VeriLens Forensics - Cloud Serverless Inference Engine (v1.0)
FastAPI Backend Interface for Vercel Serverless Hosting.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
import numpy as np
import cv2
from PIL import Image

from core.pipeline import VeriLensPipeline

app = FastAPI(
    title="VeriLens Forensics API",
    description="Multi-engine forensic detection for AI-synthesized and manipulated imagery.",
    version="1.0.0"
)

# Enable CORS for external access and Vercel routing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline in memory
pipeline = VeriLensPipeline()


def array_to_base64_png(img_array: np.ndarray) -> str:
    """Converts a numpy RGB array into a base64 encoded PNG data URI string."""
    success, encoded_image = cv2.imencode('.png', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    if not success:
        return ""
    b64_str = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "VeriLens Forensics Diagnostic Server",
        "version": "1.0.0",
        "engines_active": ["2D-FFT", "ELA-Forensics", "Metadata-Provenance"]
    }


@app.post("/api/scan")
async def scan_image(file: UploadFile = File(...)):
    # Validate MIME type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Upload an image file.")

    try:
        # Read uploaded image bytes into PIL
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Run multi-layered forensic pipeline
        report = pipeline.analyze(pil_image)

        # Encode heatmaps into base64 images for direct web visualization
        fft_heatmap_b64 = array_to_base64_png(report["visualizations"]["fft_spectrum_heatmap"])
        tampering_mask_b64 = array_to_base64_png(report["visualizations"]["tampering_mask"])

        # Construct public JSON payload
        response_data = {
            "status": "success",
            "verdict": report["verdict"],
            "confidence": report["confidence"],
            "fake_probability": report["fake_probability"],
            "diagnostics": report["diagnostics"],
            "visualizations": {
                "fft_heatmap_url": fft_heatmap_b64,
                "tampering_mask_url": tampering_mask_b64
            }
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic Pipeline Error: {str(e)}")

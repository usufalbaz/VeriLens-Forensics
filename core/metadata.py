"""
VeriLens Forensics - Metadata & Digital Provenance Engine
Inspects image metadata for AI-generation indicators, editing software,
camera-related fields, and possible C2PA/JUMBF provenance markers.

Metadata indicators are not proof of image authenticity or manipulation.
"""

import os
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


class MetadataEngine:
    """Extract metadata indicators without claiming definitive attribution."""

    AI_SIGNATURE_KEYWORDS = [
        "stable diffusion",
        "midjourney",
        "dall-e",
        "comfyui",
        "novelai",
        "automatic1111",
        "negative prompt:",
        "prompt:",
        "steps:",
        "sampler:",
    ]

    EDITING_SOFTWARE_KEYWORDS = [
        "photoshop",
        "gimp",
        "canva",
        "lightroom",
        "affinity",
        "coreldraw",
    ]

    CAMERA_TAGS = {"Make", "Model"}
    CAMERA_DETAIL_TAGS = {
        "FocalLength",
        "FNumber",
        "ISOSpeedRatings",
        "ExposureTime",
        "DateTimeOriginal",
    }

    def __init__(self):
        pass

    @staticmethod
    def _load_image(
        image_input: Union[str, os.PathLike, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Accept a path, PIL image, or supported NumPy image array."""
        if isinstance(image_input, (str, os.PathLike)):
            path = os.fspath(image_input)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image not found: {path}")
            with Image.open(path) as source:
                return source.copy()

        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                raise ValueError("Image array must not be empty.")

            if image_input.ndim not in (2, 3):
                raise ValueError("Image array must be grayscale or color.")

            if image_input.ndim == 3 and image_input.shape[2] not in (1, 3, 4):
                raise ValueError("Image array must have 1, 3, or 4 channels.")

            if image_input.dtype != np.uint8:
                if not np.issubdtype(image_input.dtype, np.number):
                    raise TypeError("Image array must contain numeric pixels.")
                if not np.all(np.isfinite(image_input)):
                    raise ValueError("Image array contains non-finite values.")
                image_input = np.clip(image_input, 0, 255).astype(np.uint8)

            array = image_input
            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]

            return Image.fromarray(array)

        raise TypeError(
            "Expected a file path, PIL Image, or NumPy image array."
        )

    @staticmethod
    def _text(value: Any, limit: int = 1000) -> str:
        """Convert metadata values to bounded, printable text."""
        try:
            text = str(value)
        except Exception:
            return ""
        return text[:limit]

    def extract_metadata(
        self,
        image_input: Union[str, os.PathLike, Image.Image, np.ndarray],
    ) -> Dict[str, Any]:
        """Return metadata findings as indicators, not proof."""

        img = self._load_image(image_input)

        raw_info: Dict[str, str] = {}
        exif_data: Dict[str, str] = {}
        ai_indicators: List[str] = []
        editing_indicators: List[str] = []
        provenance_markers: List[str] = []

        for key, value in img.info.items():
            key_text = self._text(key, 100)
            value_text = self._text(value, 1000)
            raw_info[key_text] = value_text[:300]

            searchable = f"{key_text} {value_text}".lower()

            for keyword in self.AI_SIGNATURE_KEYWORDS:
                if keyword in searchable:
                    ai_indicators.append(
                        f"Keyword '{keyword}' found in image info '{key_text}'."
                    )

            if "c2pa" in searchable or "jumbf" in searchable:
                provenance_markers.append(
                    f"Possible C2PA/JUMBF marker in image info '{key_text}'."
                )

        try:
            exif = img.getexif()

            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                value_text = self._text(value)
                exif_data[tag_name] = value_text

                searchable = f"{tag_name} {value_text}".lower()

                for keyword in self.AI_SIGNATURE_KEYWORDS:
                    if keyword in searchable:
                        ai_indicators.append(
                            f"Keyword '{keyword}' found in EXIF '{tag_name}'."
                        )

                if tag_name.lower() == "software":
                    for keyword in self.EDITING_SOFTWARE_KEYWORDS:
                        if keyword in value_text.lower():
                            editing_indicators.append(
                                f"Editing software name '{keyword}' found "
                                "in the EXIF Software field."
                            )

        except (OSError, ValueError, TypeError):
            # EXIF may be missing, malformed, or unreadable.
            pass

        ai_indicators = list(dict.fromkeys(ai_indicators))
        editing_indicators = list(dict.fromkeys(editing_indicators))
        provenance_markers = list(dict.fromkeys(provenance_markers))

        camera_tags_present = self.CAMERA_TAGS.intersection(exif_data)
        camera_details_present = self.CAMERA_DETAIL_TAGS.intersection(
            exif_data
        )

        has_camera_hardware = bool(
            camera_tags_present or camera_details_present
        )

        software_found = exif_data.get("Software")

        if ai_indicators:
            verdict = "AI_GENERATION_INDICATORS_FOUND"
        elif editing_indicators:
            verdict = "EDITING_SOFTWARE_INDICATOR_FOUND"
        elif provenance_markers:
            verdict = "PROVENANCE_MARKER_FOUND"
        elif has_camera_hardware:
            verdict = "CAMERA_METADATA_PRESENT"
        else:
            verdict = "INSUFFICIENT_METADATA"

        # A provisional review-priority score, not a probability.
        # It must not be interpreted as a percentage or proof.
        if ai_indicators:
            risk_score = 35.0
        elif editing_indicators:
            risk_score = 15.0
        else:
            risk_score = 0.0

        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "has_camera_hardware": has_camera_hardware,
            "software_detected": software_found,
            "c2pa_provenance": bool(provenance_markers),
            "ai_signatures": ai_indicators,
            "exif_summary": {
                key: exif_data[key]
                for key in list(exif_data)[:10]
            },
            "ai_generation_indicators": ai_indicators,
            "editing_indicators": editing_indicators,
            "provenance_markers": provenance_markers,
            "raw_info": raw_info,
            "analysis_scope": "Image metadata only",
            "score_type": "Uncalibrated heuristic review-priority score",
            "limitations": [
                "Metadata indicators do not prove AI generation or tampering.",
                "Metadata can be edited, copied, removed, or absent.",
                "Camera metadata does not prove that an image is authentic.",
                "C2PA/JUMBF marker detection does not validate a manifest's "
                "cryptographic integrity or trust chain.",
                "This engine does not inspect image pixels for manipulation.",
            ],
        }

"""
VeriLens Forensics - Metadata & Digital Provenance Engine (v1.0)
Extracts and inspects EXIF, C2PA manifest markers, generative prompts,
and graphic editing software signatures.
"""

from typing import Dict, Any, Union, List
import os
from PIL import Image
from PIL.ExifTags import TAGS


class MetadataEngine:
    """
    High-Speed Metadata Forensics Engine.
    Detects generative AI signatures (Stable Diffusion, Midjourney, DALL-E),
    manipulation software tags (Photoshop, GIMP), and validates camera sensor profiles.
    """
    
    # Generative AI fingerprints commonly embedded in metadata
    AI_SIGNATURE_KEYWORDS = [
        "stable diffusion", "midjourney", "dall-e", "comfyui", "novelai",
        "flux", "automatic1111", "prompt:", "negative prompt", "steps:", "sampler:"
    ]
    
    # Image editing software keywords
    EDITING_SOFTWARE_KEYWORDS = [
        "photoshop", "gimp", "canva", "lightroom", "affinity", "coreldraw"
    ]

    def __init__(self):
        pass

    def extract_metadata(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Inspects raw metadata and returns structured forensic analysis.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at path: {image_input}")
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError("Expected filepath or PIL Image instance.")

        raw_info = {}
        exif_data = {}
        ai_signatures_found = []
        software_found = None
        has_camera_hardware = False
        c2pa_detected = False

        # 1. Parse Image Info dictionary (Captures PNG Text Chunks / Parameters)
        for key, value in img.info.items():
            str_val = str(value)
            raw_info[str(key)] = str_val[:300]  # Truncate for memory efficiency
            
            # Search for AI prompt / generation parameters
            for kw in self.AI_SIGNATURE_KEYWORDS:
                if kw in str_val.lower():
                    ai_signatures_found.append(f"Param match: '{kw}' in chunk [{key}]")
            
            # Check for C2PA provenance manifests
            if "c2pa" in str_val.lower() or "jumbf" in str_val.lower():
                c2pa_detected = True

        # 2. Parse EXIF Data
        try:
            exif = img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    str_val = str(value)
                    exif_data[tag_name] = str_val
                    
                    # Software tag check
                    if tag_name.lower() == "software":
                        software_found = str_val
                        for soft_kw in self.EDITING_SOFTWARE_KEYWORDS:
                            if soft_kw in str_val.lower():
                                ai_signatures_found.append(f"Editing software signature: {str_val}")

                    # Camera hardware verification
                    if tag_name in ["Make", "Model", "FocalLength", "FNumber", "ISOSpeedRatings"]:
                        has_camera_hardware = True

                    # Search inside UserComment or descriptions
                    for kw in self.AI_SIGNATURE_KEYWORDS:
                        if kw in str_val.lower():
                            ai_signatures_found.append(f"EXIF match: '{kw}' in [{tag_name}]")
        except Exception:
            # Corrupted or missing EXIF table
            pass

        # 3. Determine Forensic Provenance Verdict
        if len(ai_signatures_found) > 0 or c2pa_detected:
            verdict = "AI_GENERATION_CONFIRMED"
            risk_score = 99.0
        elif software_found and any(s in software_found.lower() for s in self.EDITING_SOFTWARE_KEYWORDS):
            verdict = "DIGITAL_TAMPERING_SUSPECTED"
            risk_score = 75.0
        elif has_camera_hardware:
            verdict = "AUTHENTIC_CAMERA_SENSOR_DETECTED"
            risk_score = 15.0
        else:
            verdict = "STRIPPED_METADATA_ANONYMOUS"
            risk_score = 50.0

        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "has_camera_hardware": has_camera_hardware,
            "software_detected": software_found,
            "c2pa_provenance": c2pa_detected,
            "ai_signatures": list(set(ai_signatures_found)),
            "exif_summary": {k: exif_data[k] for k in list(exif_data.keys())[:10]}
        }

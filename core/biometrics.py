
"""Facial-region inspection using OpenCV Haar cascades.

Face and eye detections are visual-analysis indicators only.
They do not establish whether an image is authentic or AI-generated.
"""

from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image


ImageInput = Union[str, Path, np.ndarray, Image.Image]


class BiometricEngine:
    """Detect faces and eyes without treating them as proof of authenticity."""

    def __init__(self) -> None:
        self.face_cascade = None
        self.eye_cascade = None
        self.initialization_status = "unavailable"

        try:
            cascade_dir = getattr(cv2.data, "haarcascades", "")
            if cascade_dir:
                face_path = str(
                    Path(cascade_dir) / "haarcascade_frontalface_default.xml"
                )
                eye_path = str(
                    Path(cascade_dir) / "haarcascade_eye.xml"
                )

                face_detector = cv2.CascadeClassifier(face_path)
                eye_detector = cv2.CascadeClassifier(eye_path)

                if not face_detector.empty():
                    self.face_cascade = face_detector

                if not eye_detector.empty():
                    self.eye_cascade = eye_detector

                if self.face_cascade is not None:
                    self.initialization_status = "ready"
        except Exception:
            self.face_cascade = None
            self.eye_cascade = None
            self.initialization_status = "unavailable"

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """Convert supported image inputs to an 8-bit BGR image."""

        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Could not read the image file.")

            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            if image.shape[2] == 3:
                return image

            raise ValueError("Unsupported image channel count.")

        if isinstance(image_input, Image.Image):
            rgb = np.asarray(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            image = np.asarray(image_input)

            if image.size == 0:
                raise ValueError("Image array is empty.")

            if not np.issubdtype(image.dtype, np.number):
                raise TypeError("Image array must contain numeric values.")

            if not np.isfinite(image).all():
                raise ValueError("Image array contains non-finite values.")

            image = np.clip(image, 0, 255).astype(np.uint8)

            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if image.ndim != 3:
                raise ValueError("Image array must be 2D or 3D.")

            if image.shape[2] == 1:
                return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            raise ValueError("Unsupported image channel count.")

        raise TypeError("Unsupported image input type.")

    def analyze_faces(self, image_input: ImageInput) -> Dict[str, Any]:
        """Detect facial regions and report limited visual indicators."""

        base_result = {
            "faces_detected": 0,
            "biometric_verdict": "INCONCLUSIVE",
            "biometric_risk_score": 0.0,
            "score_type": "uncalibrated_heuristic",
            "face_details": [],
            "analysis_scope": "face_and_eye_detection_only",
            "limitations": [
                "Face and eye detection cannot determine image authenticity.",
                "Eye brightness differences are affected by lighting and pose.",
                "Blur can result from focus, motion, compression, or resizing.",
                "No validated deepfake classifier is implemented in this module.",
                "A zero score does not mean the image is authentic.",
            ],
        }

        if self.face_cascade is None:
            base_result["biometric_verdict"] = "MODULE_UNAVAILABLE"
            return base_result

        image = self._load_image(image_input)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        face_details = []

        for index, (x, y, width, height) in enumerate(faces):
            face_gray = gray[y:y + height, x:x + width]
            eyes_found = 0

            if self.eye_cascade is not None and face_gray.size:
                eyes = self.eye_cascade.detectMultiScale(
                    face_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(15, 15),
                )
                eyes_found = len(eyes)

            face_details.append({
                "face_id": index + 1,
                "bounding_box": [
                    int(x), int(y), int(width), int(height)
                ],
                "eyes_found": int(eyes_found),
                "detection_status": "face_detected",
            })

        base_result["faces_detected"] = len(face_details)
        base_result["face_details"] = face_details

        if face_details:
            base_result["biometric_verdict"] = "FACE_DETECTED_INCONCLUSIVE"
        else:
            base_result["biometric_verdict"] = "NO_FACES_DETECTED"

        return base_result

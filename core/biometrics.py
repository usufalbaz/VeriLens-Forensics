
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image, ImageOps


ImageInput = Union[str, Path, np.ndarray, Image.Image]


class BiometricEngine:
    """
    Facial-region inspection using OpenCV Haar cascades.

    Face and eye detections are visual-analysis indicators only.
    They do not establish whether an image is authentic or AI-generated.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        face_min_neighbors: int = 5,
        eye_min_neighbors: int = 4,
        min_face_size: int = 60,
        min_eye_size: int = 15,
    ) -> None:

        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0.")

        if not isinstance(face_min_neighbors, int) or face_min_neighbors < 1:
            raise ValueError("face_min_neighbors must be a positive integer.")

        if not isinstance(eye_min_neighbors, int) or eye_min_neighbors < 1:
            raise ValueError("eye_min_neighbors must be a positive integer.")

        if not isinstance(min_face_size, int) or min_face_size < 1:
            raise ValueError("min_face_size must be a positive integer.")

        if not isinstance(min_eye_size, int) or min_eye_size < 1:
            raise ValueError("min_eye_size must be a positive integer.")

        self.scale_factor = float(scale_factor)
        self.face_min_neighbors = face_min_neighbors
        self.eye_min_neighbors = eye_min_neighbors
        self.min_face_size = min_face_size
        self.min_eye_size = min_eye_size

        self.face_cascade = None
        self.eye_cascade = None

        self.initialization_status = "unavailable"
        self.initialization_errors = []

        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize OpenCV Haar cascade classifiers."""

        try:
            cascade_dir = getattr(
                getattr(cv2, "data", None),
                "haarcascades",
                None,
            )

            if not cascade_dir:
                self.initialization_errors.append(
                    "OpenCV Haar cascade directory is unavailable."
                )
                return

            cascade_dir = Path(cascade_dir)

            face_path = cascade_dir / "haarcascade_frontalface_default.xml"
            eye_path = cascade_dir / "haarcascade_eye.xml"

            if face_path.is_file():
                detector = cv2.CascadeClassifier(str(face_path))

                if not detector.empty():
                    self.face_cascade = detector
                else:
                    self.initialization_errors.append(
                        "Face cascade could not be loaded."
                    )
            else:
                self.initialization_errors.append(
                    "Face cascade file was not found."
                )

            if eye_path.is_file():
                detector = cv2.CascadeClassifier(str(eye_path))

                if not detector.empty():
                    self.eye_cascade = detector
                else:
                    self.initialization_errors.append(
                        "Eye cascade could not be loaded."
                    )
            else:
                self.initialization_errors.append(
                    "Eye cascade file was not found."
                )

            if self.face_cascade is not None:
                self.initialization_status = "ready"
            else:
                self.initialization_status = "unavailable"

        except Exception as exc:
            self.face_cascade = None
            self.eye_cascade = None
            self.initialization_status = "unavailable"
            self.initialization_errors.append(
                f"Detector initialization failed: {type(exc).__name__}"
            )

    @staticmethod
    def _convert_array_to_uint8(array: np.ndarray) -> np.ndarray:
        """Normalize supported numeric arrays to 8-bit image data."""

        if array.size == 0:
            raise ValueError("Image array is empty.")

        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("Image array must contain numeric values.")

        if not np.isfinite(array).all():
            raise ValueError("Image array contains non-finite values.")

        if np.issubdtype(array.dtype, np.floating):
            minimum = float(array.min())
            maximum = float(array.max())

            if minimum >= 0.0 and maximum <= 1.0:
                array = array * 255.0

        elif array.dtype == np.uint16:
            array = array.astype(np.float32) * (255.0 / 65535.0)

        return np.clip(array, 0, 255).astype(np.uint8)

    @classmethod
    def _load_image(cls, image_input: ImageInput) -> np.ndarray:
        """Convert supported inputs to an independent 8-bit BGR image."""

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)

            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")

            with Image.open(path) as source:
                corrected = ImageOps.exif_transpose(source)
                rgb = np.asarray(
                    corrected.convert("RGB"),
                    dtype=np.uint8,
                ).copy()

            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, Image.Image):
            corrected = ImageOps.exif_transpose(image_input)
            rgb = np.asarray(
                corrected.convert("RGB"),
                dtype=np.uint8,
            ).copy()

            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            array = cls._convert_array_to_uint8(
                np.asarray(image_input)
            )

            if array.ndim == 2:
                return cv2.cvtColor(
                    array,
                    cv2.COLOR_GRAY2BGR,
                )

            if array.ndim != 3:
                raise ValueError("Image array must be 2D or 3D.")

            channels = array.shape[2]

            if channels == 1:
                return cv2.cvtColor(
                    array[:, :, 0],
                    cv2.COLOR_GRAY2BGR,
                )

            if channels == 3:
                return cv2.cvtColor(
                    array,
                    cv2.COLOR_RGB2BGR,
                )

            if channels == 4:
                return cv2.cvtColor(
                    array,
                    cv2.COLOR_RGBA2BGR,
                )

            raise ValueError("Unsupported image channel count.")

        raise TypeError("Unsupported image input type.")

    @staticmethod
    def _measure_region_quality(
        gray_region: np.ndarray,
    ) -> Dict[str, Any]:
        """Measure descriptive image-quality indicators in a face region."""

        if gray_region.size == 0:
            return {
                "brightness_mean": 0.0,
                "brightness_std": 0.0,
                "contrast_std": 0.0,
                "blur_variance": 0.0,
                "measurement_status": "empty_region",
            }

        gray_float = gray_region.astype(np.float32)

        brightness_mean = float(np.mean(gray_float))
        brightness_std = float(np.std(gray_float))

        laplacian = cv2.Laplacian(
            gray_region,
            cv2.CV_64F,
        )

        blur_variance = float(np.var(laplacian))

        return {
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "contrast_std": brightness_std,
            "blur_variance": blur_variance,
            "measurement_status": "measured",
            "interpretation": {
                "brightness": "Mean grayscale intensity from 0 to 255.",
                "contrast": "Standard deviation of grayscale intensity.",
                "blur": (
                    "Variance of the Laplacian; lower values can indicate "
                    "reduced sharpness but are not proof of blur."
                ),
            },
        }

    def _detect_eyes(
        self,
        face_gray: np.ndarray,
        face_x: int,
        face_y: int,
    ) -> list:
        """Detect eye candidates within a detected face region."""

        if self.eye_cascade is None or face_gray.size == 0:
            return []

        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.eye_min_neighbors,
            minSize=(
                self.min_eye_size,
                self.min_eye_size,
            ),
        )

        eye_details = []

        face_height, face_width = face_gray.shape[:2]

        for index, (x, y, width, height) in enumerate(eyes):
            eye_region = face_gray[
                y:y + height,
                x:x + width,
            ]

            eye_details.append({
                "eye_id": index + 1,
                "bounding_box": [
                    int(face_x + x),
                    int(face_y + y),
                    int(width),
                    int(height),
                ],
                "relative_position": {
                    "x_center": float(
                        (x + width / 2) / max(face_width, 1)
                    ),
                    "y_center": float(
                        (y + height / 2) / max(face_height, 1)
                    ),
                },
                "quality": self._measure_region_quality(
                    eye_region
                ),
            })

        return eye_details

    def _analyze_single_face(
        self,
        gray: np.ndarray,
        face_box: tuple,
        face_id: int,
    ) -> Dict[str, Any]:
        """Analyze one detected facial region."""

        x, y, width, height = map(int, face_box)

        image_height, image_width = gray.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_width, x + width)
        y2 = min(image_height, y + height)

        face_gray = gray[y1:y2, x1:x2]

        face_quality = self._measure_region_quality(
            face_gray
        )

        eyes = self._detect_eyes(
            face_gray,
            x1,
            y1,
        )

        return {
            "face_id": face_id,
            "bounding_box": [
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ],
            "normalized_bounding_box": {
                "x": float(x1 / image_width),
                "y": float(y1 / image_height),
                "width": float((x2 - x1) / image_width),
                "height": float((y2 - y1) / image_height),
            },
            "eyes_found": len(eyes),
            "eye_details": eyes,
            "face_quality": face_quality,
            "detection_status": "face_detected",
        }

    def analyze_faces(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """Detect and inspect facial regions."""

        base_result = {
            "faces_detected": 0,
            "biometric_verdict": "INCONCLUSIVE",
            "biometric_risk_score": 0.0,
            "score_type": "uncalibrated_heuristic",
            "face_details": [],
            "analysis_scope": "face_and_eye_detection_only",
            "detector_status": self.initialization_status,
            "limitations": [
                "Face and eye detection cannot determine image authenticity.",
                "Haar cascades can miss faces depending on pose, lighting, and image quality.",
                "Eye detections may include false positives.",
                "Brightness and contrast depend on lighting and exposure.",
                "Low Laplacian variance can result from focus, motion, compression, or resizing.",
                "No validated deepfake classifier is implemented in this module.",
                "A zero score does not mean the image is authentic.",
            ],
        }

        if self.face_cascade is None:
            base_result["biometric_verdict"] = "MODULE_UNAVAILABLE"
            base_result["initialization_errors"] = (
                self.initialization_errors
            )
            return base_result

        image = self._load_image(image_input)

        if image.size == 0:
            raise ValueError("Loaded image is empty.")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        image_height, image_width = gray.shape[:2]

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.face_min_neighbors,
            minSize=(
                self.min_face_size,
                self.min_face_size,
            ),
        )

        face_details = []

        for index, face_box in enumerate(faces):
            details = self._analyze_single_face(
                gray,
                face_box,
                index + 1,
            )

            face_details.append(details)

        base_result.update({
            "faces_detected": len(face_details),
            "face_details": face_details,
            "image_metadata": {
                "width": int(image_width),
                "height": int(image_height),
                "channels": 3,
                "color_space": "BGR",
            },
            "configuration": {
                "scale_factor": self.scale_factor,
                "face_min_neighbors": self.face_min_neighbors,
                "eye_min_neighbors": self.eye_min_neighbors,
                "min_face_size": self.min_face_size,
                "min_eye_size": self.min_eye_size,
            },
        })

        if face_details:
            base_result["biometric_verdict"] = (
                "FACE_DETECTED_INCONCLUSIVE"
            )
        else:
            base_result["biometric_verdict"] = (
                "NO_FACES_DETECTED"
            )

        return base_result

    def analyze(
        self,
        image_input: ImageInput,
    ) -> Dict[str, Any]:
        """Compatibility alias for face analysis."""

        return self.analyze_faces(image_input)

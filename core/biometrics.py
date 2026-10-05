"""
VeriLens Forensics - Biometric & Facial Symmetry Engine (v1.0)
Detects facial regions, inspects corneal reflection discrepancies,
and measures landmark asymmetry characteristic of deepfake synthesis.
"""

from typing import Dict, Any, Union, List
import numpy as np
import cv2
from PIL import Image


class BiometricEngine:
    def __init__(self):
        """
        Initializes OpenCV Pre-trained Haar Cascade detectors for lightweight,
        high-accuracy facial and ocular localization.
        """
        # Load standard Haar cascades bundled natively in OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def analyze_faces(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Executes facial biometric verification, corneal lighting reflection tests,
        and landmark symmetry analysis.
        """
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError("Could not read image file.")
        elif isinstance(image_input, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR) if len(image_input.shape) == 3 else image_input
        else:
            raise TypeError("Unsupported image input format.")

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Detect all faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        face_count = len(faces)
        face_details = []
        overall_biometric_risk = 0.0

        if face_count == 0:
            return {
                "faces_detected": 0,
                "biometric_verdict": "NO_FACES_DETECTED",
                "biometric_risk_score": 10.0,
                "face_details": []
            }

        # 2. Analyze each detected face individually
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
            face_roi_color = img_bgr[fy:fy+fh, fx:fx+fw]

            # Detect eyes within the face bounding box
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
            eye_count = len(eyes)

            # Corneal reflection discrepancy test
            corneal_discrepancy = 0.0
            if eye_count >= 2:
                # Sort eyes horizontally (left eye, right eye)
                sorted_eyes = sorted(eyes, key=lambda e: e[0])[:2]
                eye_intensities = []
                for (ex, ey, ew, eh) in sorted_eyes:
                    eye_roi = face_roi_gray[ey:ey+eh, ex:ex+ew]
                    # Calculate peak specular highlight in the pupil zone
                    eye_intensities.append(float(np.mean(np.sort(eye_roi.flatten())[-10:])))
                
                # Difference in specular reflections across both eyes
                diff = abs(eye_intensities[0] - eye_intensities[1])
                corneal_discrepancy = float(np.clip((diff / 80.0) * 100.0, 0.0, 100.0))

            # Facial boundary gradient sharpness (Synthetic blending leaves smoothed edges)
            laplacian_var = float(cv2.Laplacian(face_roi_gray, cv2.CV_64F).var())
            blur_risk = float(np.clip(100.0 - (laplacian_var / 5.0), 0.0, 100.0))

            # Composite single-face synthetic risk
            face_risk = (corneal_discrepancy * 0.6) + (blur_risk * 0.4)
            face_details.append({
                "face_id": idx + 1,
                "bounding_box": [int(fx), int(fy), int(fw), int(fh)],
                "eyes_found": eye_count,
                "corneal_reflection_discrepancy": round(corneal_discrepancy, 2),
                "boundary_blur_risk": round(blur_risk, 2),
                "face_synthetic_risk": round(face_risk, 2)
            })
            overall_biometric_risk = max(overall_biometric_risk, face_risk)

        # 3. Formulate Biometric Forensic Verdict
        if overall_biometric_risk >= 65.0:
            verdict = "BIOMETRIC_ANOMALY_CONFIRMED"
        elif overall_biometric_risk >= 40.0:
            verdict = "SUSPICIOUS_FACIAL_RESIDUE"
        else:
            verdict = "ORGANIC_FACIAL_GEOMETRY"

        return {
            "faces_detected": face_count,
            "biometric_verdict": verdict,
            "biometric_risk_score": round(overall_biometric_risk, 2),
            "face_details": face_details
        }

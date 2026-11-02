import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image, ImageOps


class DeepfakeEngine:
    def __init__(
        self,
        model_path: Optional[str] = None,
        image_size: int = 224,
        fake_class_index: int = 1,
        threshold: float = 0.5,
    ):
        self.model_path = model_path or os.getenv("DEEPFAKE_MODEL_PATH")
        self.image_size = image_size
        self.fake_class_index = fake_class_index
        self.threshold = threshold

        self.model = None
        self.device = None
        self.torch = None
        self.initialized = False
        self.error = None

    def initialize(self) -> bool:
        if self.initialized:
            return True

        if not self.model_path:
            self.error = "No trained model path configured."
            return False

        if not Path(self.model_path).is_file():
            self.error = "Model file does not exist."
            return False

        try:
            import torch

            self.torch = torch
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self.model = torch.jit.load(
                self.model_path,
                map_location=self.device,
            )

            self.model.eval()
            self.initialized = True
            self.error = None
            return True

        except ImportError:
            self.error = "PyTorch is not installed in this runtime."
        except Exception as exc:
            self.error = f"Model initialization failed: {exc}"

        return False

    def _load_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Image.Image:
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")

        elif isinstance(image_input, Image.Image):
            image = ImageOps.exif_transpose(image_input).convert("RGB")

        elif isinstance(image_input, np.ndarray):
            array = np.asarray(image_input)

            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)

            if array.ndim != 3:
                raise ValueError("Unsupported image dimensions.")

            if array.shape[2] == 4:
                array = array[:, :, :3]

            if array.shape[2] != 3:
                raise ValueError("Expected RGB or grayscale image.")

            if np.issubdtype(array.dtype, np.floating):
                array = np.nan_to_num(array)
                if array.max() <= 1.0:
                    array = array * 255.0

            array = np.clip(array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array).convert("RGB")

        else:
            raise TypeError("Unsupported image input type.")

        if image.width < 32 or image.height < 32:
            raise ValueError("Image dimensions are too small.")

        return image

    def _preprocess(self, image: Image.Image):
        torch = self.torch

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(image, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)

        return torch.from_numpy(array).to(self.device)

    def analyze(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> Dict[str, Any]:

        if not self.initialize():
            return {
                "engine": "deepfake",
                "status": "unavailable",
                "available": False,
                "prediction": None,
                "fake_probability": None,
                "real_probability": None,
                "error": self.error,
            }

        try:
            image = self._load_image(image_input)
            tensor = self._preprocess(image)

            with self.torch.inference_mode():
                output = self.model(tensor)

                if isinstance(output, (tuple, list)):
                    output = output[0]

                if isinstance(output, dict):
                    if "logits" in output:
                        output = output["logits"]
                    else:
                        raise ValueError(
                            "Model output dictionary must contain 'logits'."
                        )

                output = output.detach().float().reshape(1, -1)

                if output.shape[1] == 1:
                    fake_probability = self.torch.sigmoid(output)[0, 0]
                else:
                    probabilities = self.torch.softmax(output, dim=1)

                    if not 0 <= self.fake_class_index < output.shape[1]:
                        raise ValueError("Invalid fake class index.")

                    fake_probability = probabilities[
                        0, self.fake_class_index
                    ]

                fake_probability = float(fake_probability.item())
                real_probability = 1.0 - fake_probability

            prediction = (
                "fake"
                if fake_probability >= self.threshold
                else "real"
            )

            return {
                "engine": "deepfake",
                "status": "success",
                "available": True,
                "prediction": prediction,
                "fake_probability": round(fake_probability, 6),
                "real_probability": round(real_probability, 6),
                "threshold": self.threshold,
                "model_type": "torchscript",
                "limitations": [
                    "Predictions depend on the model's training data.",
                    "This result is not proof of image authenticity.",
                    "The model must be validated on unseen datasets.",
                ],
            }

        except Exception as exc:
            return {
                "engine": "deepfake",
                "status": "error",
                "available": True,
                "prediction": None,
                "fake_probability": None,
                "real_probability": None,
                "error": str(exc),
            }

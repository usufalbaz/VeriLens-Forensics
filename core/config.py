
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")

    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class Settings:
    app_name: str = "VeriLens Forensics"
    app_version: str = "2.0.0"
    environment: str = "development"

    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 25_000_000
    allowed_image_formats: tuple[str, ...] = (
        "JPEG",
        "PNG",
        "WEBP",
        "BMP",
        "TIFF",
    )

    enable_metadata: bool = True
    enable_forensics: bool = True
    enable_frequency: bool = True
    enable_biometrics: bool = True
    enable_deepfake: bool = True

    deepfake_model_path: str | None = None
    deepfake_threshold: float = 0.5

    inference_service_url: str | None = None
    inference_service_timeout: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        environment = os.getenv(
            "VERILENS_ENV",
            "development",
        ).strip().lower()

        if environment not in {
            "development",
            "testing",
            "production",
        }:
            raise ValueError("Invalid VERILENS_ENV")

        threshold = _env_float(
            "VERILENS_DEEPFAKE_THRESHOLD",
            0.5,
        )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "VERILENS_DEEPFAKE_THRESHOLD must be between 0 and 1"
            )

        model_path = os.getenv("VERILENS_MODEL_PATH")
        service_url = os.getenv("VERILENS_INFERENCE_URL")

        return cls(
            environment=environment,
            max_upload_bytes=_env_int(
                "VERILENS_MAX_UPLOAD_BYTES",
                10 * 1024 * 1024,
            ),
            max_image_pixels=_env_int(
                "VERILENS_MAX_IMAGE_PIXELS",
                25_000_000,
            ),
            enable_metadata=_env_bool(
                "VERILENS_ENABLE_METADATA",
                True,
            ),
            enable_forensics=_env_bool(
                "VERILENS_ENABLE_FORENSICS",
                True,
            ),
            enable_frequency=_env_bool(
                "VERILENS_ENABLE_FREQUENCY",
                True,
            ),
            enable_biometrics=_env_bool(
                "VERILENS_ENABLE_BIOMETRICS",
                True,
            ),
            enable_deepfake=_env_bool(
                "VERILENS_ENABLE_DEEPFAKE",
                True,
            ),
            deepfake_model_path=model_path or None,
            deepfake_threshold=threshold,
            inference_service_url=service_url or None,
            inference_service_timeout=_env_int(
                "VERILENS_INFERENCE_TIMEOUT",
                60,
            ),
        )


settings = Settings.from_env()

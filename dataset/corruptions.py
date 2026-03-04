"""
Image corruption functions using albumentations.
13 corruption types with 5 severity levels, matching ImageNet-C benchmark.
"""
from typing import Callable

import numpy as np
from PIL import Image
import cv2
import albumentations as A



# ---------------------------------------------------------------------------
# 13 corruption types
# ---------------------------------------------------------------------------

CORRUPTION_TYPES: list[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


def _build_transform(corruption_type: str, severity: int) -> A.BasicTransform:
    """Return an albumentations transform for the given corruption and severity.

    Parameters
    ----------
    corruption_type : str
        One of :data:`CORRUPTION_TYPES`.
    severity : int
        Severity level in [1, 5].  Higher = stronger corruption.
    """
    assert 1 <= severity <= 5, f"severity must be in [1,5], got {severity}"

    match corruption_type:
        # ---- Noise --------------------------------------------------------
        case "gaussian_noise":
            # var_limit controls variance range; we pick a fixed value per severity
            var_map = {1: (8, 12), 2: (15, 25), 3: (30, 45), 4: (50, 70), 5: (80, 110)}
            return A.GaussNoise(var_limit=var_map[severity], p=1.0)

        case "shot_noise":
            # scale_range: lower = more noise (fewer effective photons)
            scale_map = {1: (0.8, 1.0), 2: (0.5, 0.8), 3: (0.3, 0.5),
                         4: (0.15, 0.3), 5: (0.05, 0.15)}
            return A.ShotNoise(scale_range=scale_map[severity], p=1.0)

        case "impulse_noise":
            # amount: fraction of pixels replaced with salt/pepper
            amount_map = {1: (0.01, 0.02), 2: (0.03, 0.05), 3: (0.06, 0.10),
                          4: (0.12, 0.18), 5: (0.20, 0.30)}
            return A.SaltAndPepper(amount=amount_map[severity], p=1.0)

        # ---- Blur ---------------------------------------------------------
        case "defocus_blur":
            radius_map = {1: (2, 3), 2: (3, 5), 3: (5, 7), 4: (7, 9), 5: (9, 12)}
            return A.Defocus(radius=radius_map[severity], alias_blur=(0.1, 0.5), p=1.0)

        case "motion_blur":
            blur_map = {1: (3, 5), 2: (5, 9), 3: (9, 13), 4: (13, 17), 5: (17, 23)}
            return A.MotionBlur(blur_limit=blur_map[severity], p=1.0)

        case "zoom_blur":
            # max_factor: how much zoom distortion
            factor_map = {1: (1.01, 1.06), 2: (1.06, 1.11), 3: (1.11, 1.16),
                          4: (1.16, 1.21), 5: (1.21, 1.31)}
            return A.ZoomBlur(max_factor=factor_map[severity], p=1.0)

        # ---- Weather ------------------------------------------------------
        case "snow":
            coeff_map = {1: (0.1, 0.2), 2: (0.2, 0.3), 3: (0.3, 0.45),
                         4: (0.45, 0.6), 5: (0.6, 0.8)}
            return A.RandomSnow(
                snow_point_range=(0.1, 0.3),
                brightness_coeff=sum(coeff_map[severity]) / 2,
                p=1.0,
            )

        case "fog":
            coeff_map = {1: (0.1, 0.2), 2: (0.2, 0.35), 3: (0.35, 0.5),
                         4: (0.5, 0.65), 5: (0.65, 0.85)}
            return A.RandomFog(
                fog_coef_range=coeff_map[severity],
                alpha_coef=0.1,
                p=1.0,
            )

        # ---- Digital / Photometric ----------------------------------------
        case "brightness":
            limit_map = {1: (0.05, 0.15), 2: (0.15, 0.25), 3: (0.25, 0.35),
                         4: (0.35, 0.50), 5: (0.50, 0.70)}
            return A.RandomBrightnessContrast(
                brightness_limit=limit_map[severity],
                contrast_limit=(0, 0),
                p=1.0,
            )

        case "contrast":
            limit_map = {1: (-0.15, -0.05), 2: (-0.25, -0.15), 3: (-0.40, -0.25),
                         4: (-0.55, -0.40), 5: (-0.75, -0.55)}
            return A.RandomBrightnessContrast(
                brightness_limit=(0, 0),
                contrast_limit=limit_map[severity],
                p=1.0,
            )

        case "elastic_transform":
            alpha_map = {1: 20, 2: 50, 3: 100, 4: 200, 5: 400}
            sigma_map = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}
            return A.ElasticTransform(
                alpha=alpha_map[severity],
                sigma=sigma_map[severity],
                p=1.0,
            )

        case "pixelate":
            # Downscale + nearest neighbour upscale -> pixelation effect
            scale_map = {1: (0.7, 0.8), 2: (0.5, 0.7), 3: (0.35, 0.5),
                         4: (0.2, 0.35), 5: (0.1, 0.2)}
            return A.Downscale(
                scale_range=scale_map[severity],
                interpolation_pair={
                    "downscale": cv2.INTER_NEAREST,
                    "upscale": cv2.INTER_NEAREST,
                },
                p=1.0,
            )

        case "jpeg_compression":
            # quality_range: lower = more compression artifacts
            quality_map = {1: (60, 80), 2: (40, 60), 3: (25, 40),
                           4: (12, 25), 5: (2, 12)}
            return A.ImageCompression(
                quality_range=quality_map[severity],
                p=1.0,
            )

        case _:
            raise ValueError(
                f"Unknown corruption_type={corruption_type!r}. "
                f"Choose from: {CORRUPTION_TYPES}"
            )


def get_corruption_func(corruption_type: str,
                        severity: int) -> Callable[[Image.Image], Image.Image]:
    """Return a callable  ``PIL.Image -> PIL.Image``  that applies the
    requested corruption at the given severity.

    This is a drop-in replacement for the old ``imagenet_c.corrupt`` usage.
    """
    transform = _build_transform(corruption_type, severity)

    def _corrupt(img: Image.Image) -> Image.Image:
        arr = np.asarray(img)  # (H, W, C) uint8
        result = transform(image=arr)["image"]
        return Image.fromarray(result)

    return _corrupt
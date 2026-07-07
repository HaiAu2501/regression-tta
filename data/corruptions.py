from collections.abc import Callable
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import albumentations as A
import cv2
import numpy as np
from PIL import Image


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
    assert 1 <= severity <= 5, f"severity must be in [1,5], got {severity}"

    match corruption_type:
        case "gaussian_noise":
            var_map = {1: (20, 30), 2: (40, 60), 3: (70, 100),
                       4: (120, 170), 5: (180, 250)}
            return A.GaussNoise(var_limit=var_map[severity], p=1.0)
        case "shot_noise":
            scale_map = {1: (0.5, 0.7), 2: (0.25, 0.5),
                         3: (0.1, 0.25), 4: (0.05, 0.1),
                         5: (0.02, 0.05)}
            return A.ShotNoise(scale_range=scale_map[severity], p=1.0)
        case "impulse_noise":
            amount_map = {1: (0.03, 0.05), 2: (0.06, 0.10),
                          3: (0.12, 0.18), 4: (0.20, 0.30),
                          5: (0.35, 0.50)}
            return A.SaltAndPepper(amount=amount_map[severity], p=1.0)
        case "defocus_blur":
            radius_map = {1: (3, 5), 2: (5, 8), 3: (8, 12),
                          4: (12, 16), 5: (16, 22)}
            return A.Defocus(radius=radius_map[severity],
                             alias_blur=(0.1, 0.5), p=1.0)
        case "motion_blur":
            blur_map = {1: (7, 11), 2: (11, 17), 3: (17, 23),
                        4: (23, 31), 5: (31, 41)}
            return A.MotionBlur(blur_limit=blur_map[severity], p=1.0)
        case "zoom_blur":
            factor_map = {1: (1.05, 1.11), 2: (1.11, 1.18),
                          3: (1.18, 1.26), 4: (1.26, 1.35),
                          5: (1.35, 1.50)}
            return A.ZoomBlur(max_factor=factor_map[severity], p=1.0)
        case "snow":
            snow_point_map = {1: (0.2, 0.35), 2: (0.35, 0.50),
                              3: (0.50, 0.65), 4: (0.65, 0.80),
                              5: (0.80, 1.0)}
            bright_map = {1: 1.5, 2: 1.8, 3: 2.1, 4: 2.5, 5: 3.0}
            return A.RandomSnow(snow_point_range=snow_point_map[severity],
                                brightness_coeff=bright_map[severity],
                                p=1.0)
        case "fog":
            coeff_map = {1: (0.2, 0.35), 2: (0.35, 0.50),
                         3: (0.50, 0.65), 4: (0.65, 0.80),
                         5: (0.80, 0.95)}
            return A.RandomFog(fog_coef_range=coeff_map[severity],
                               alpha_coef=0.1, p=1.0)
        case "brightness":
            limit_map = {1: (0.10, 0.20), 2: (0.20, 0.30),
                         3: (0.30, 0.45), 4: (0.45, 0.60),
                         5: (0.60, 0.75)}
            return A.RandomBrightnessContrast(brightness_limit=limit_map[severity],
                                              contrast_limit=(0, 0), p=1.0)
        case "contrast":
            limit_map = {1: (-0.30, -0.20), 2: (-0.45, -0.30),
                         3: (-0.60, -0.45), 4: (-0.75, -0.60),
                         5: (-0.90, -0.75)}
            return A.RandomBrightnessContrast(brightness_limit=(0, 0),
                                              contrast_limit=limit_map[severity],
                                              p=1.0)
        case "elastic_transform":
            alpha_map = {1: 50, 2: 150, 3: 300, 4: 500, 5: 800}
            sigma_map = {1: 5, 2: 6, 3: 7, 4: 8, 5: 10}
            return A.ElasticTransform(alpha=alpha_map[severity],
                                      sigma=sigma_map[severity], p=1.0)
        case "pixelate":
            scale_map = {1: (0.5, 0.6), 2: (0.35, 0.5),
                         3: (0.2, 0.35), 4: (0.1, 0.2),
                         5: (0.05, 0.1)}
            return A.Downscale(
                scale_range=scale_map[severity],
                interpolation_pair={
                    "downscale": cv2.INTER_NEAREST,
                    "upscale": cv2.INTER_NEAREST,
                },
                p=1.0,
            )
        case "jpeg_compression":
            quality_map = {1: (40, 60), 2: (25, 40), 3: (12, 25),
                           4: (5, 12), 5: (1, 5)}
            return A.ImageCompression(quality_range=quality_map[severity],
                                      p=1.0)
        case _:
            raise ValueError(
                f"Unknown corruption_type={corruption_type!r}. "
                f"Choose from: {CORRUPTION_TYPES}"
            )


def get_corruption_func(corruption_type: str,
                        severity: int) -> Callable[[Image.Image], Image.Image]:
    transform = _build_transform(corruption_type, severity)

    def corrupt(img: Image.Image) -> Image.Image:
        arr = np.asarray(img)
        result = transform(image=arr)["image"]
        return Image.fromarray(result)

    return corrupt

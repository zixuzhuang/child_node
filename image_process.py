import numpy as np
import torch
from .utils.image import normalize

class ImageNormalize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_ratio": (
                    "FLOAT",
                    {
                        "default": 99.5,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "min_ratio": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0,  # Minimum value
                        "max": 100,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_normailze"
    CATEGORY = "child/ImageProcessing"

    def image_normailze(self, image, max_ratio, min_ratio):
        image, t_max, t_min = normalize(image, max_ratio / 100, min_ratio / 100)
        return (image,)






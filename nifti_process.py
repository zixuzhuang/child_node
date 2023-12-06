import concurrent.futures
import hashlib
import json
import math
import os
import random
import shutil
import time

import cv2
import imageio
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image, ImageOps
from .utils.image import normalize
import folder_paths
from comfy.cli_args import args

input_dir = "input/nifti"


class LoadNifti:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "nifti_file": (sorted(files),),
                "orientation": (
                    "STRING",
                    {"default": "PIL", "options": ["PIL", "LIP", "LPI"]},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGES",)
    CATEGORY = "child/Nifti"
    FUNCTION = "execute"

    def execute(self, nifti_file, orientation):
        data = sitk.ReadImage(os.path.join(input_dir, nifti_file))
        data = sitk.DICOMOrient(data, orientation)
        image = sitk.GetArrayFromImage(data).astype(np.float32)
        image = torch.tensor(image)
        print(image.shape)

        return [image]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


class Nifti2Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_normailze"
    CATEGORY = "child/Nifti"

    def image_normailze(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        return (image,)


def normalize(image, img_max=0.995, img_min=0.005):
    assert type(image) is torch.Tensor or np.ndarray
    if type(image) is torch.Tensor:
        t_max = torch.quantile(image, img_max)
        t_min = torch.quantile(image, img_min)
    else:
        t_max = np.percentile(image, img_max * 100)
        t_min = np.percentile(image, img_min * 100)
    image = (image - t_min) / (t_max - t_min)
    image[image > 1] = 1
    image[image < 0] = 0
    return image, t_max, t_min

# from .face_roop import Face_Roop, Roop_loader
from .image_process import ImageNormalize
from .nifti_process import LoadNifti, Nifti2Image


NODE_CLASS_MAPPINGS = {
    "ImageNormalize": ImageNormalize,
    "NiftiLoader": LoadNifti,
    "Nifti2Image": Nifti2Image,
}

__all__ = ["NODE_CLASS_MAPPINGS"]

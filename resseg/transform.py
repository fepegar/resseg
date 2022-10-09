import numpy as np
import nibabel as nib
import torchio as tio

from constants import IMAGE_NAME
from constants import TO_MNI


RESSEG_LANDMARKS = np.array([
      0.        ,   0.31331614,   0.61505419,   0.76732501,
      0.98887953,   1.71169384,   3.21741126,  13.06931455,
     32.70817796,  40.87807389,  47.83508873,  63.4408591 ,
    100.
])
NUM_LEVELS_UNET = 3


def get_preprocessing_transform(
    input_path,
    mni_transform_path=None,
    interpolation='bspline',
    tolerance=0.1,
):
    hist_std = tio.HistogramStandardization({IMAGE_NAME: RESSEG_LANDMARKS})
    preprocess_transforms = [
        tio.ToCanonical(),
        hist_std,
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ]
    zooms = nib.load(input_path).header.get_zooms()
    pixdim = np.array(zooms)
    diff_to_1_iso = np.abs(pixdim - 1)
    if np.any(diff_to_1_iso > tolerance) or mni_transform_path is not None:
        kwargs = {'image_interpolation': interpolation}
        if mni_transform_path is not None:
            kwargs['pre_affine_name'] = TO_MNI
            kwargs['target'] = tio.datasets.Colin27().t1.path
        resample_transform = tio.Resample(**kwargs)
        preprocess_transforms.append(resample_transform)
    target = int(2 ** NUM_LEVELS_UNET)
    ensure_shape = tio.EnsureShapeMultiple(target, method='crop')
    preprocess_transforms.append(ensure_shape)
    preprocess_transform = tio.Compose(preprocess_transforms)
    return preprocess_transform

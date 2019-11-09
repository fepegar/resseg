from pathlib import Path
from typing import Union, Optional
import numpy as np
import nibabel as nib


def load(
        path: Union[str, Path],
        mmap: bool = True,
        ) -> nib.Nifti1Image:
    nii = nib.load(str(path), mmap=mmap)
    return nii

def save(
        data: np.ndarray,
        path: Union[str, Path],
        affine: Optional[np.ndarray] = None,
        rgb: bool = False,
        ) -> None:
    nii = nib.Nifti1Image(data, affine)
    nii.header['qform_code'] = 1
    nii.header['sform_code'] = 0
    if rgb:
        nii.header.set_intent('vector')
    nib.save(nii, str(path))


def get_data(path: Union[str, Path]) -> np.ndarray:
    data = load(path).get_data()
    if isinstance(data, np.memmap):
        data = np.array(data)
    return data


read = load
write = save

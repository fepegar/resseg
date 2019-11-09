import numpy as np
import nibabel as nib
from utils.nifti import load, save


def flip_lr(input_path, output_path):
    # Asume R-- or L-- (e.g. RAS or LPS)
    nii = load(input_path, mmap=False)
    data = nii.get_data()
    data = data[::-1, ...]
    save(data, affine=nii.affine, path=output_path)


def mean_image(input_paths, output_path):
    niis = [load(fp) for fp in input_paths]
    arrays = [nii.get_data() for nii in niis]
    mean = np.mean(arrays, axis=0)
    save(mean, path=output_path, affine=niis[0].affine)


def binarize_probabilities(input_path, output_path):
    nii = load(input_path, mmap=False)
    data = nii.get_data() > 0.5
    data = data.astype(np.uint8)
    save(data, affine=nii.affine, path=output_path)

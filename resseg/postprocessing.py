import numpy as np
import nibabel as nib
import SimpleITK as sitk
from .nifti import load, save


def keep_largest_cc(input_path, output_path):
    image = sitk.ReadImage(str(input_path))
    connected_components = sitk.ConnectedComponent(image)
    labeled_cc = sitk.RelabelComponent(connected_components)
    largest_cc = labeled_cc == 1
    sitk.WriteImage(largest_cc, str(output_path))

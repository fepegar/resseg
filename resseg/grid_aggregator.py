import numpy as np
import nibabel as nib


class GridAggregator:
    """
    Adapted from NiftyNet
    """
    def __init__(self, reference_image_path, window_border):
        self.reference_nii = nib.load(str(reference_image_path))
        self.window_border = window_border
        self.output_array = np.full(
            self.reference_nii.shape,
            fill_value=0,
            dtype=np.float32,
        )

    @staticmethod
    def crop_batch(windows, location, border=None):
        if not border:
            return windows, location
        location = location.astype(np.int)
        batch_shape = windows.shape
        spatial_shape = batch_shape[2:]  # ignore batch and channels dim
        num_dimensions = 3
        for idx in range(num_dimensions):
            location[:, idx] = location[:, idx] + border[idx]
            location[:, idx + 3] = location[:, idx + 3] - border[idx]
        if np.any(location < 0):
            return windows, location

        cropped_shape = np.max(location[:, 3:6] - location[:, 0:3], axis=0)
        diff = spatial_shape - cropped_shape
        left = np.floor(diff / 2).astype(np.int)
        i_ini, j_ini, k_ini = left
        i_fin, j_fin, k_fin = left + cropped_shape
        if np.any(left < 0):
            raise ValueError
        batch = windows[
            :,  # batch dimension
            :,  # channels dimension
            i_ini:i_fin,
            j_ini:j_fin,
            k_ini:k_fin,
        ]
        return batch, location

    def add_batch(self, windows, locations):
        windows = windows.cpu()
        location_init = np.copy(locations)
        init_ones = np.ones_like(windows)
        windows, _ = self.crop_batch(
            windows, location_init,
            self.window_border,
        )
        location_init = np.copy(locations)
        _, locations = self.crop_batch(
            init_ones,
            location_init,
            self.window_border,
        )
        for window, location in zip(windows, locations):
            window = window.squeeze()
            i_ini, j_ini, k_ini, i_fin, j_fin, k_fin = location
            self.output_array[i_ini:i_fin, j_ini:j_fin, k_ini:k_fin] = window

    def save_current_image(self, output_path, output_probabilities=False):
        if not output_probabilities:
            self.output_array = self.output_array.astype(np.uint16)
        nii = nib.Nifti1Image(self.output_array, self.reference_nii.affine)
        nib.save(nii, str(output_path))

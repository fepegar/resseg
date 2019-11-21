from tempfile import NamedTemporaryFile

import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .grid_sampler import GridSampler
from .grid_aggregator import GridAggregator
from .postprocessing import binarize_probabilities, flip_lr, mean_image, keep_largest_cc


def to_tuple(value, n=3):
    if isinstance(value, str):
        split = value.split(',')
        value = int(split[0]) if len(split) == 1 else tuple(int(n) for n in split)
    try:
        iter(value)
    except TypeError:
        value = n * (value,)
    return value


def get_device():
    # pylint: disable=no-member
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_divisible(array, n):
    shape = np.array(array.shape)
    mod = shape % n
    pad = n - mod
    zeros = 0, 0, 0
    pad_width = list(zip(zeros, pad))
    array = np.pad(array, pad_width)
    return array


def segment_resection(
        input_path,
        model,
        window_size,
        window_border,
        output_path,
        batch_size=None,
        show_progress=True,
        flip=True,
        binarize=True,
        whole_image=False,
        postprocess=True,
        ):

    run_inference(
        input_path,
        model,
        output_path=output_path,
        window_size=window_size,
        window_border=window_border,
        batch_size=batch_size,
        show_progress=show_progress,
        whole_image=whole_image,
        postprocess=postprocess,
    )

    if flip:
        with NamedTemporaryFile(suffix='.nii') as output_temp:
            with NamedTemporaryFile(suffix='.nii') as input_temp:
                flip_lr(input_path, input_temp.name)
                run_inference(
                    input_temp.name,
                    model,
                    output_path=output_temp.name,
                    window_size=window_size,
                    window_border=window_border,
                    batch_size=batch_size,
                    show_progress=show_progress,
                    whole_image=whole_image,
                    postprocess=postprocess,
                )
            flip_lr(output_temp.name, output_temp.name)
            paths = output_path, output_temp.name
            mean_image(paths, output_path)

    if binarize:
        binarize_probabilities(output_path, output_path)

    if postprocess:
        keep_largest_cc(output_path, output_path)


def run_inference(
        image_path,
        model,
        output_path,
        window_size,
        window_border=None,
        batch_size=None,
        show_progress=True,
        whole_image=False,
        ):

    if whole_image:
        nii = nib.load(str(image_path))
        array = nii.get_data()
        array = pad_divisible(array, 8)  # HARDCODE UNET 3 LEVELS
        batch_image = array[np.newaxis, np.newaxis, ...].astype(np.float32)
        batch_image = torch.from_numpy(batch_image)
        batch = dict(
            image=batch_image,
        )
        batches = [batch]
    else:
        window_border = 1 if window_border is None else window_border
        batch_size = 1 if batch_size is None else batch_size
        window_size = to_tuple(window_size)
        window_border = to_tuple(window_border)

        sampler = GridSampler(
            image_path, window_size, window_border, dtype=np.float32)
        aggregator = GridAggregator(image_path, window_border)
        batches = DataLoader(sampler, batch_size=batch_size)

    device = get_device()
    print('Using device', device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        progress = tqdm(batches) if show_progress else batches
        for batch in progress:
            input_tensor = batch['image'].to(device)
            logits = model(input_tensor)
            probabilities = logits.softmax(dim=1)
            foreground = probabilities[:, 1:, ...]
            outputs = foreground
            if not whole_image:
                locations = batch['location']
                aggregator.add_batch(outputs, locations)

    if whole_image:
        array = outputs.cpu().numpy().squeeze()

        # In case it was padded
        si, sj, sk = nii.shape
        array = array[:si, :sj, :sk]

        output_nii = nib.Nifti1Image(array, nii.affine)
        output_nii.header['qform_code'] = 1
        output_nii.header['sform_code'] = 0
        output_nii.to_filename(str(output_path))
    else:
        aggregator.save_current_image(
            output_path,
            output_probabilities=True,
        )

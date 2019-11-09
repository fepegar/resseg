from tempfile import NamedTemporaryFile

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .grid_sampler import GridSampler
from .grid_aggregator import GridAggregator
from .postprocessing import binarize_probabilities, flip_lr, mean_image


def to_tuple(value):
    try:
        iter(value)
    except TypeError:
        value = 3 * (value,)
    return value


def get_device():
    # pylint: disable=no-member
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        ):

    run_inference(
        input_path,
        model,
        output_path=output_path,
        window_size=window_size,
        window_border=window_border,
        batch_size=batch_size,
        show_progress=show_progress,
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
                )
            flip_lr(output_temp.name, output_temp.name)
            paths = output_path, output_temp.name
            mean_image(paths, output_path)

    if binarize:
        binarize_probabilities(output_path, output_path)


def run_inference(
        image_path,
        model,
        output_path,
        window_size,
        window_border=None,
        batch_size=None,
        show_progress=True,
        ):
    window_border = 1 if window_border is None else window_border
    batch_size = 1 if batch_size is None else batch_size
    window_size = to_tuple(window_size)
    window_border = to_tuple(window_border)

    sampler = GridSampler(image_path, window_size, window_border)
    aggregator = GridAggregator(image_path, window_border)
    loader = DataLoader(sampler, batch_size=batch_size)

    device = get_device()
    print('Using device', device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        progress = tqdm(loader) if show_progress else loader
        for batch in progress:
            input_tensor = batch['image'].to(device)
            locations = batch['location']
            logits = model(input_tensor)
            probabilities = logits.softmax(dim=1)
            foreground = probabilities[:, 1:, ...]
            outputs = foreground
            aggregator.add_batch(outputs, locations)

    aggregator.save_current_image(
        output_path,
        output_probabilities=True,
    )

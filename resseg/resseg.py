# -*- coding: utf-8 -*-

"""Main module."""

import torch
from .inference import segment_resection


def resseg(
        input_path,
        output_path,
        iterations,
        num_workers,
        batch_size,
        postprocess=True,
        ):
    repo = 'fepegar/resseg'
    model_name = 'ressegnet'
    model = torch.hub.load(repo, model_name)
    segment_resection(
        input_path,
        model,
        output_path,
        iterations,
        num_workers,
        batch_size,
        postprocess=postprocess,
    )

# -*- coding: utf-8 -*-

"""Main module."""

import torch
from .inference import segment_resection


def resseg(input_path, output_path, window_size, window_border, batch_size, whole_image=False):
    repo = 'fepegar/resseg'
    model_name = 'resseg'
    print(torch.hub.help(repo, model_name))
    model = torch.hub.load(repo, model_name, pretrained=True)
    segment_resection(
        input_path,
        model,
        window_size,
        window_border,
        output_path,
        batch_size=batch_size,
        whole_image=whole_image,
    )

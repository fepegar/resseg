# -*- coding: utf-8 -*-

"""Main module."""

import torch
from resseg.inference import segment_resection


def resseg(
        input_path,
        output_path,
        tta_iterations,
        interpolation,
        num_workers,
        postprocess=True,
        mni_transform_path=None,
        ):
    repo = 'fepegar/resseg'
    model_name = 'ressegnet'
    model = torch.hub.load(repo, model_name)
    segment_resection(
        input_path,
        model,
        output_path=output_path,
        tta_iterations=tta_iterations,
        interpolation=interpolation,
        num_workers=num_workers,
        postprocess=postprocess,
        mni_transform_path=mni_transform_path,
    )

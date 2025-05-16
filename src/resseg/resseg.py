"""Main module."""

from pathlib import Path

import torch

from resseg.inference import segment_resection


def resseg(
    input_path: Path,
    output_path: Path,
    tta_iterations: int,
    interpolation: str,
    num_workers: int,
    *,
    postprocess: bool = True,
    mni_transform_path: Path | None = None,
):
    repo = "fepegar/resseg"
    model_name = "ressegnet"
    model: torch.nn.Module = torch.hub.load(repo, model_name, trust_repo=True)  # type: ignore[reportAssignmentType]
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

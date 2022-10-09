from pathlib import Path

import torch
import numpy as np
import torchio as tio
from tqdm import tqdm

from .constants import FROM_MNI
from .constants import IMAGE_NAME
from .constants import TO_MNI
from .utils import get_device
from .transform import get_preprocessing_transform


def segment_resection(
        input_path,
        model,
        output_path=None,
        tta_iterations=0,
        interpolation='bspline',
        num_workers=0,
        show_progress=True,
        binarize=True,
        postprocess=True,
        mni_transform_path=None,
        ):
    dataset = get_dataset(
        input_path,
        tta_iterations,
        interpolation,
        mni_transform_path=mni_transform_path,
    )

    device = get_device()
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )
    all_results = []
    for subjects_list_batch in tqdm(loader, disable=not show_progress):
        tensors = [subject[IMAGE_NAME][tio.DATA] for subject in subjects_list_batch]
        inputs = torch.stack(tensors).float().to(device)
        with torch.cuda.amp.autocast():
            try:
                probs = model(inputs).softmax(dim=1)[:, 1:].cpu()  # discard background
            except Exception as e:
                print(e)
                raise
        iterable = list(zip(subjects_list_batch, probs))
        for subject, prob in tqdm(iterable, leave=False, unit='subject'):
            subject.image.set_data(prob)
            subject_back = subject.apply_inverse_transform(warn=False, image_interpolation='linear')
            all_results.append(subject_back.image.data)
    result = torch.stack(all_results)
    mean_prob = result.mean(dim=0)
    if binarize:
        mean_prob = (mean_prob >= 0.5).byte()
        class_ = tio.LabelMap
    else:
        class_ = tio.ScalarImage
    image_kwargs = {
        'tensor': mean_prob,
        'affine': subject_back[IMAGE_NAME].affine,
    }
    resample_kwargs = {
        'target': input_path,
        'image_interpolation': interpolation,
    }
    if mni_transform_path is not None:
        to_mni = tio.io.read_matrix(mni_transform_path)
        from_mni = np.linalg.inv(to_mni)
        image_kwargs[FROM_MNI] = from_mni
        resample_kwargs['pre_affine_name'] = FROM_MNI
    image = class_(**image_kwargs)
    if postprocess:
        image = tio.KeepLargestComponent()(image)
    resample = tio.Resample(**resample_kwargs)
    image_native = resample(image)
    if output_path is None:
        input_path = Path(input_path)
        split = input_path.name.split('.')
        stem = split[0]
        exts_string = '.'.join(split[1:])
        output_path = input_path.parent / f'{stem}_seg.{exts_string}'
    image_native.save(output_path)


def get_dataset(
        input_path,
        tta_iterations=0,
        interpolation='bspline',
        mni_transform_path=None,
        ):
    if mni_transform_path is None:
        image = tio.ScalarImage(input_path)
    else:
        affine = tio.io.read_matrix(mni_transform_path)
        image = tio.ScalarImage(input_path, **{TO_MNI: affine})
    subject = tio.Subject({IMAGE_NAME: image})

    preprocess_transform = get_preprocessing_transform(
        input_path,
        mni_transform_path=mni_transform_path,
        interpolation=interpolation,
    )

    no_aug_dataset = tio.SubjectsDataset([subject], transform=preprocess_transform)

    aug_subjects = tta_iterations * [subject]
    if not aug_subjects:
        return no_aug_dataset
    augment_transform = tio.Compose((
        preprocess_transform,
        tio.RandomFlip(),
        tio.RandomAffine(image_interpolation=interpolation),
    ))
    aug_dataset = tio.SubjectsDataset(aug_subjects, transform=augment_transform)
    dataset = torch.utils.data.ConcatDataset((no_aug_dataset, aug_dataset))
    return dataset

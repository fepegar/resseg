import warnings

import torch
import numpy as np
import nibabel as nib
import torchio as tio
from tqdm import tqdm

import resseg.utils


IMAGE_NAME = 'image'


def segment_resection(
        input_path,
        model,
        output_path,
        tta_iterations,
        interpolation='bspline',
        num_workers=0,
        show_progress=True,
        binarize=True,
        postprocess=True,  # ignore for now
        ):
    dataset = get_dataset(input_path, tta_iterations, interpolation)

    device = resseg.utils.get_device()
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
        subjects = batch[IMAGE_NAME][tio.DATA].to(device)
        tensors = [subject[IMAGE_NAME][tio.DATA] for subject in subjects_list_batch]
        inputs = torch.stack(tensors).float().to(device)
        with torch.cuda.amp.autocast():
            try:
                probs = model(inputs).softmax(dim=1)[:, 1:]  # discard background
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
    image = class_(tensor=mean_prob, affine=subject_back[IMAGE_NAME].affine)
    resample = tio.Resample(input_path, image_interpolation=interpolation)
    image_native = resample(image)
    image_native.save(output_path)


def get_dataset(input_path, tta_iterations, interpolation, tolerance=0.1):
    subject = tio.Subject({IMAGE_NAME: tio.ScalarImage(input_path)})
    zooms = nib.load(input_path).header.get_zooms()
    pixdim = np.array(zooms)
    diff_to_1_iso = np.abs(pixdim - 1)
    resample = False
    if np.any(diff_to_1_iso > tolerance):
        message = (
            f'Pixel spacing is too far from 1 mm isotropic: {zooms}.'
            'Images will be resampled before inference.'
        )
        warnings.warn(message)
        resample = True
    preprocess_transforms = [
        tio.ToCanonical(),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ]
    if resample:
        resample_transform = tio.Resample(image_interpolation=interpolation)
        preprocess_transforms.append(resample_transform)
    preprocess_transform = tio.Compose(preprocess_transforms)
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
    dataset = torch.utils.data.ConcatDataset((no_aug_dataset, augment_transform))
    return dataset

from pathlib import Path

import torch
import torchio as tio
from tqdm import tqdm

from .utils import get_device
from .inference import get_dataset, IMAGE_NAME


def get_module(model, part, level, conv_layer):
    assert part in ('encoder', 'decoder')
    assert conv_layer in (0, 1)
    if part == 'encoder':
        assert level in (0, 1, 2)
        if level < 2:
            blocks = model.encoder.encoding_blocks
            conv_block = blocks[level]
        elif level == 2:
            conv_block = model.bottom_block
    elif part == 'decoder':
        assert level in (0, 1)
        blocks = model.decoder.decoding_blocks
        conv_block = blocks[-1 - level]
    if conv_layer == 0:
        return conv_block.conv1.conv_layer
    elif conv_layer == 1:
        return conv_block.conv2.conv_layer


def get_all_modules(model):
    modules = []
    for part in ('encoder', 'decoder'):
        for level in (0, 1, 2):
            for conv_layer in (0, 1):
                try:
                    args = part, level, conv_layer
                    module = get_module(model, *args)
                    modules.append((args, module))
                except AssertionError:
                    pass
    return modules


global activation
def hook_fn(module, input, output):
    global activation
    activation = output


activation = {}
def get_activation(args):
    def hook(model, input, output):
        activation[args] = output.detach()
    return hook


def save_feature_maps(input_path, output_dir):
    torch.set_grad_enabled(False)
    output_dir = Path(output_dir).expanduser().absolute()
    output_dir.mkdir(exist_ok=True, parents=True)
    device = get_device()
    repo = 'fepegar/resseg'
    model_name = 'ressegnet'
    model = torch.hub.load(repo, model_name)
    model.to(device)
    model.eval()
    hooks = []
    for args, module in get_all_modules(model):
        hook = module.register_forward_hook(get_activation(args))
    preprocessed_subject = get_dataset(input_path)[0]
    image = preprocessed_subject[IMAGE_NAME]
    inputs = image.data.unsqueeze(0).float().to(device)
    with torch.cuda.amp.autocast():
        model(inputs)

    downsampled = [image]
    for _ in range(2):
        target = torch.Tensor(downsampled[-1].spacing) * 2
        target = tuple(target.tolist())
        transform = tio.Resample(target, image_interpolation='nearest')
        downsampled_image = transform(downsampled[-1])
        downsampled.append(downsampled_image)
    for args, features in tqdm(activation.items()):
        part, level, conv_layer = args
        affine = downsampled[level].affine
        for i, feature_map in enumerate(tqdm(features[0], leave=False)):
            features_image = tio.ScalarImage(
                tensor=feature_map.unsqueeze(0).cpu().float(),
                affine=affine,
            )
            name = f'{part}_level_{level}_layer_{conv_layer}_feature_{i}.nii.gz'
            path = output_dir / name
            features_image.save(path)

from pathlib import Path

import torch
from unet import UNet


WEIGHTS_URL = 'https://github.com/fepegar/resseg/raw/master/self_semi_37-b571f7ba.pth'


def ressegnet(pretrained: bool = True, progress: bool = True):
    model = UNet(
        in_channels=1,
        out_classes=2,
        dimensions=3,
        num_encoding_blocks=3,
        out_channels_first_layer=8,
        normalization='batch',
        pooling_type='max',
        padding=True,
        padding_mode='replicate',
        residual=False,
        initial_dilation=1,
        activation='PReLU',
        upsampling_type='linear',
        dropout=0,
        monte_carlo_dropout=0.5,
    )
    if pretrained:
        repo_dir = Path(__file__).parent.parent
        weights_path = repo_dir / 'self_semi_37-b571f7ba.pth'
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model

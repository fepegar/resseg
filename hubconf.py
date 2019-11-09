dependencies = ['unet', 'torch']

import json
import inspect
import urllib.request
import torch
from unet import UNet3D


def resseg(*args, pretrained=False, **kwargs):
    if pretrained:
        repo_dir = 'https://github.com/fepegar/resseg/raw/master'
        config_url = '{}/2019-10-28_18-44-16_config.json'.format(repo_dir)
        config = Config(config_url)
        model = config.get_unet()
        time = '2019-10-28_18-44-16'
        url = '{}/{}_epoch_190_state_dict-4e20e838.pth'.format(repo_dir, time)
        state_dict = torch.hub.load_state_dict_from_url(
            url, progress=False, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        model = UNet3D(*args, **kwargs)
    return model


class Config:
    def __init__(self, config_url):
        self.config = self.read_config(config_url)

    def read_config(self, config_url):
        with urllib.request.urlopen(config_url) as url:
            config = json.loads(url.read().decode())
        return config

    def get_model_args(self, model_class, model_args):
        model_class_arg_spec = inspect.getfullargspec(model_class)
        model_class_args = model_class_arg_spec.args
        for config_key, value in self.config.items():
            if config_key in model_class_args:
                model_args[config_key] = value

    def get_unet(self):
        from unet import UNet
        model_class = UNet
        model_args = dict(
            in_channels=1,
            out_classes=2,
            dimensions=3,
        )
        self.get_model_args(model_class, model_args)
        model = model_class(**model_args)
        return model

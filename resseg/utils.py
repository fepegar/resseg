import torch


def get_device():
    # pylint: disable=no-member
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

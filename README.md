# RESSEG

Automatic segmentation of postoperative brain resection cavities from magnetic resonance images (MRI) using a convolutional neural network (CNN) trained with [PyTorch](https://pytorch.org/) 1.7.1.

## Installation

It's recommended to use [`conda`](https://docs.conda.io/en/latest/miniconda.html) and [install your desired PyTorch version](https://pytorch.org/get-started/locally/) before
installing `resseg`.
A 6-GB GPU is large enough to segment an image in the MNI space.

```shell
conda create -n resseg python=3.8 ipython -y && conda activate resseg  # recommended
pip install resseg
```

## Usage

Below are two examples of cavity segmentation for tumor and epilepsy surgery. The epilepsy example includes registration to the [MNI space](https://www.lead-dbs.org/about-the-mni-spaces/).
Both examples can be run online using Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fepegar/resseg/blob/master/RESSEG.ipynb)

### BITE

Example using an image from the
[Brain Images of Tumors for Evaluation database (BITE)](http://nist.mni.mcgill.ca/?page_id=672).

```shell
BITE=`resseg-download bite`
resseg $BITE -o bite_seg.nii.gz
```

![Resection cavity segmented on an image from BITE](screenshots/bite.png)

### EPISURG

Example using an image from the [EPISURG dataset](https://doi.org/10.5522/04/9996158.v1).
Segmentation works best when images are in the MNI space, so `resseg` includes a tool
for this purpose (requires [antspyx](https://antspyx.readthedocs.io/en/latest/?badge=latest)).

```shell
pip install antspyx
EPISURG=`resseg-download episurg`
resseg-mni $EPISURG -t episurg_to_mni.tfm
resseg $EPISURG -o episurg_seg.nii.gz -t episurg_to_mni.tfm
```

![Resection cavity segmented on an image from EPISURG](screenshots/episurg.png)

## Trained model

The trained model can be used without installing `resseg`, but you'll need to install `unet` first:

```shell
pip install unet==0.7.7
```

Then, in Python:

```python
import torch
repo = 'fepegar/resseg'
model_name = 'ressegnet'
model = torch.hub.load(repo, model_name, pretrained=True)
```

## Credit

If you use this library for your research, please cite our MICCAI 2020 paper:

[F. Pérez-García, R. Rodionov, A. Alim-Marvasti, R. Sparks, J. S. Duncan and S. Ourselin. *Simulation of Brain Resection for Cavity Segmentation Using Self-Supervised and Semi-Supervised Learning*](https://link.springer.com/chapter/10.1007%2F978-3-030-59716-0_12).

[[Preprint on arXiv](https://arxiv.org/abs/2006.15693)]

And the [EPISURG dataset](https://doi.org/10.5522/04/9996158.v1), which was used to train the model:

[Pérez-García, Fernando; Rodionov, Roman; Alim-Marvasti, Ali; Sparks, Rachel; Duncan, John; Ourselin, Sebastien (2020): *EPISURG: a dataset of postoperative magnetic resonance images (MRI) for quantitative analysis of resection neurosurgery for refractory epilepsy*. University College London. Dataset. https://doi.org/10.5522/04/9996158.v1](https://doi.org/10.5522/04/9996158.v1)

## See also

- [`resector`](https://github.com/fepegar/resector) was used to simulate brain resections during training
- [TorchIO](http://torchio.rtfd.io/) was also used extensively. Both `resseg` and `resector` require this library.

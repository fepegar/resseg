# RESSEG

Automatic segmentation of postoperative brain resection cavities.

## Installation

It's recommended to use `conda` and install your desired PyTorch version before
installing `resseg`.
A 6-GB GPU is large enough to segment an image in the MNI space.

```shell
conda create -n resseg python=3.8 ipython -y && conda activate resseg  # recommended
pip install resseg
```

## Usage examples

### BITE

Example using an image from the
[Brain Images of Tumors for Evaluation database (BITE)](http://nist.mni.mcgill.ca/?page_id=672).

```shell
BITE=`resseg-download bite`
resseg $BITE -o bite_seg.nii.gz
tiohd --plot bite_seg.nii.gz
```

### EPISURG

Example using an image from the [EPISURG dataset](https://doi.org/10.5522/04/9996158.v1).
Segmentation works best when images are in the MNI space, so `resseg` includes a tool
for this purpose (requires [ANTsPy](https://antspyx.readthedocs.io/en/latest/?badge=latest)).


```shell
EPISURG=`resseg-download episurg`
resseg-mni $EPISURG -t episurg_to_mni.tfm
resseg $EPISURG -o episurg_seg.nii.gz -t episurg_to_mni.tfm
tiohd --plot episurg_seg.nii.gz
```

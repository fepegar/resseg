import sys
import click

from .resseg import INPUT_FILE_TYPE, OUTPUT_FILE_TYPE


@click.command()
@click.argument('input-path', type=INPUT_FILE_TYPE)
@click.option(
    '--transform-path', '-t',
    type=OUTPUT_FILE_TYPE,
    help='Path to the output affine transform. The recommended extension is .tfm'
)
@click.option(
    '--resampled-image-path', '-r',
    type=OUTPUT_FILE_TYPE,
    help='Path to the output image, resampled using linear interpolation'
)
def main(input_path, transform_path, resampled_image_path):
    if resampled_image_path is None and transform_path is None:
        raise ValueError('You must provide the path to at least one output')
    try:
        import ants
    except ImportError as e:
        message = 'Install ANTS for registration: pip install antspyx'
        raise ModuleNotFoundError(message) from e
    import numpy as np
    import torchio as tio
    reference = ants.image_read(str(tio.datasets.Colin27().t1.path))
    floating = ants.image_read(input_path)
    results = ants.registration(reference, floating, type_of_transform='Affine')
    if transform_path is not None:
        transform = ants.read_transform(results['fwdtransforms'][0])
        affine = np.eye(4)
        affine[:3] = transform.parameters.reshape(4, 3).T
        affine = tio.io._from_itk_convention(affine)
        tio.io.write_matrix(affine, transform_path)
    if resampled_image_path is not None:
        ants.image_write(results['warpedmovout'], resampled_image_path)
    return 0


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover

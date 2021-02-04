import os
import sys
import urllib

import click


@click.command()
@click.argument(
    'dataset',
    type=click.Choice(['EPISURG', 'BITE'], case_sensitive=False),
)
def main(dataset):
    if dataset == 'EPISURG':
        url = 'https://github.com/fepegar/resseg/raw/master/sample_data/sub-0089_postop-t1mri-1_u8.nii.gz'
    elif dataset == 'BITE':
        url = 'https://github.com/fepegar/resseg/raw/master/sample_data/04_postop_mri_bite_u8.nii.gz'
    print(download_file(url))
    return 0


def download_file(url):
    base_name = os.path.basename(url)
    urllib.request.urlretrieve(url, filename=base_name)[0]
    return base_name


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover

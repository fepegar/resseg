# -*- coding: utf-8 -*-

"""Console script for resseg."""
import sys
import click


INPUT_FILE_TYPE = click.Path(exists=True, dir_okay=False)
OUTPUT_FILE_TYPE = click.Path(dir_okay=True)

@click.command()
@click.argument('input-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output-dir', type=click.Path(dir_okay=True))
def main(input_path, output_dir):
    """Console script to visualize feauture maps in ressegnet."""
    from resseg.features import save_feature_maps
    save_feature_maps(input_path, output_dir)
    return 0


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover

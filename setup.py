#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'click',
    'torch>=1.6',
    'torchio',
    'unet==0.7.7',
]

setup(
    author="Fernando Perez-Garcia",
    author_email='fernando.perezgarcia.17@ucl.ac.uk',
    python_requires='>2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="Automatic segmentation of epilepsy neurosurgery resection cavity.",
    entry_points={
        'console_scripts': [
            'resseg=resseg.cli.resseg:main',
            'resseg-mni=resseg.cli.resseg_mni:main',
            'resseg-download=resseg.cli.resseg_download:main',
            'resseg-features=resseg.cli.resseg_feature_maps:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='resseg',
    name='resseg',
    packages=find_packages(include=['resseg', 'resseg.*']),
    test_suite='tests',
    url='https://github.com/fepegar/resseg',
    version='0.3.6',
    zip_safe=False,
)

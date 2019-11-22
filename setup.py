#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
    'nibabel',
    'numpy',
    'torch>=1.1',
    'tqdm',
    'unet',
    # 'utils @ https://github.com/fepegar/utils/archive/master.zip#egg=utils'
]

setup_requirements = []

test_requirements = []


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
        'Programming Language :: Python :: 3.6',
    ],
    description="Automatic segmentation of epilepsy neurosurgery resection cavity.",
    entry_points={
        'console_scripts': [
            'resseg=resseg.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='resseg',
    name='resseg',
    packages=find_packages(include=['resseg', 'resseg.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fepegar/resseg',
    version='0.2.1',
    zip_safe=False,
)

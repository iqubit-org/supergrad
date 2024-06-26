#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: `Ziang Wang`
Date: 2022-07-12 09:23:17
LastEditTime: 2022-07-12 09:23:17
"""
import importlib
import os
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
project_name = 'supergrad'


def load_version_module(pkg_path):
    spec = importlib.util.spec_from_file_location(
        'version', os.path.join(pkg_path, 'version.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_version_module = load_version_module(project_name)
__version__ = _version_module._get_version_for_build()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='supergrad',
    version=__version__,
    author='IQUBIT',
    url='https://github.com/iqubit-org/supergrad',
    description='SuperGrad differentiable Hamiltonian simulator',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples", "tests"]),
    package_data={},
    python_requires='>=3.9',
    platforms='any',
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'jax', 'dm_haiku', 'jaxopt', 'networkx',
        'tqdm', 'rich'
    ],
    extras_require={
        # Used for benchmark
        'benchmark': [
            'qiskit_dynamics', 'qutip', 'pytest', 'pytest-benchmark',
            'scqubits', 'psutil'
        ],
        'docs': ['qutip', 'qutip-qip'],
        # Used for generate paper
        'latex': ['pylatex'],
    },
    license='Apache-2.0',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: `Ziang Wang`
Date: 2022-07-12 09:23:17
LastEditTime: 2022-07-12 09:23:17
"""

from setuptools import setup, find_namespace_packages
from os import path

here = path.abspath(path.dirname(__file__))

MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='supergrad',
      version=VERSION,
      author='Alibaba Quantum Laboratory',
      url='https://github.com/allegro0132/supergrad',
      description='SuperGrad differentiable Hamiltonian simulator',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_namespace_packages(include=["supergrad.*"]),
      platforms='any',
      install_requires=[
          'numpy', 'scipy', 'matplotlib', 'jax', 'dm_haiku', 'jaxopt', 'optax',
          'pint', 'networkx', 'rich', 'pytest', 'pylatex', 'tqdm'
      ],
      include_package_data=True,
      package_data={})

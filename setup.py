#!/usr/bin/env python
import os
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install

with open("requirements.txt") as f:
    DEPENDENCIES = f.read().splitlines()

setup(
    name="deep_rl_asset_allocation",
    packages=find_packages(),
    version="0.0.0",
    description="Package for assest allocation using deep RL.",
    author="Aidan Keaveny",
    license="MIT",
    install_requires=DEPENDENCIES,
    python_requires=">=3.8",
)

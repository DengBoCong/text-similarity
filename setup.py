#! -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

setup(
    name='text-sim',
    version='0.0.1',
    description='an elegant text-sim',
    long_description='text-sim: https://github.com/DengBoCong/sim',
    long_description_content_type="text/markdown",
    license='MIT License',
    url='https://github.com/DengBoCong/sim',
    author='DengBoCong',
    author_email='bocongdeng@gmail.com',
    install_requires=['scikit-learn>=0.23.2', 'numpy>=1.19.2'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

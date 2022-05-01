#! -*- coding: utf-8 -*-

import codecs
import pathlib
from setuptools import setup
from setuptools import find_packages

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
with codecs.open("requirements.txt", "r", "utf8") as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(
    name="text-sim",
    version="1.0.7",
    description="Chinese text similarity calculation package of Tensorflow/Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/DengBoCong/text-similarity",
    author="DengBoCong",
    author_email="bocongdeng@gmail.com",
    install_requires=install_requires,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="similarity, tensorflow, pytorch, classification",
    project_urls={
        "Bug Reports": "https://github.com/DengBoCong/text-similarity/issues",
        "Funding": "https://pypi.org/project/text-sim/",
        "Source": "https://github.com/DengBoCong/text-similarity",
    },
)

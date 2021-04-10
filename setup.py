#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='sentence2vec',
    version='0.0.1',
    description='an elegant sentence2vec',
    long_description='sentence2vec: https://github.com/DengBoCong/sentence2vec',
    license='MIT License',
    url='https://github.com/DengBoCong/sentence2vec',
    author='DengBoCong',
    author_email='bocongdeng@gmail.com',
    install_requires=['tensorflow>=2.2.0', 'scikit-learn>=0.23.2'],
    packages=find_packages()
)

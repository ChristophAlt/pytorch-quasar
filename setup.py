#!/usr/bin/env python 3.6
import os
import io
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get(
                "encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('README.md') as f:
    long_description = f.read()

VERSION = find_version('quasar', '__init__.py')

setup_info = dict(
    # Metadata
    name='pytorch-quasar',
    version=VERSION,
    author='Christoph Alt',
    author_email='chrizz.alt@gmail.com',
    url='https://github.com/ChristophAlt/pytorch-quasar',
    description='NLP train and experimentation tools for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='BSD',
    install_requires=['numpy', 'pytorch-nlp', 'visdom', 'pytorch-ignite'],
    dependency_links=['git+https://github.com/ChristophAlt/ignite.git@master#egg=pytorch-ignite-0.0.1a'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='pytorch nlp text torchtext torchnlp visdom experiments',
    python_requires='>=3.6',

    project_urls={
        'Source': 'https://github.com/ChristophAlt/pytorch-quasar/',
    },

    # Package info
    packages=find_packages(exclude=['build_tools', 'docs', 'tests']),
    zip_safe=True)

setup(**setup_info)

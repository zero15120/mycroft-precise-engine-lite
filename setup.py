#!/usr/bin/env python3
# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup

from precise import __version__

setup(
    name='mycroft-precise',
    version=__version__,
    license='Apache-2.0',
    author='Matthew Scholefield',
    author_email='matthew.scholefield@mycroft.ai',
    description='Mycroft Precise Wake Word Listener',
    long_description='View more info at `the GitHub page '
                     '<https://github.com/mycroftai/mycroft-precise#mycroft-precise>`_',
    url='http://github.com/MycroftAI/mycroft-precise',
    keywords='wakeword keyword wake word listener sound',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=[
        'precise',
        'precise.scripts'
    ],
    entry_points={
        'console_scripts': [
            'precise-engine=precise.scripts.engine:main',
        ]
    },
    install_requires=[
        'tflite_runtime',
        'numpy',
        'sonopy',
        'pyaudio',
        'typing',
        'prettyparse>=1.1.0',
        'precise-runner',
        'attrs'
    ]
)
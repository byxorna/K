from setuptools import setup, find_packages

setup(
    name='k',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm', # TODO: remove these dependencies -> environment.yaml
    ],
)

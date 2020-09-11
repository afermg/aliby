from setuptools import setup

setup(
    name='pipeline-core',
    version='0.1.1-dev',
    packages=['core'],
    url='',
    license='',
    author='Diane Adjavon',
    author_email='diane.adjavon@ed.ac.uk',
    description='Core utilities for microscopy pipeline',
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=[
        'numpy',
        'tqdm',
        'scikit-image==0.16.2',
        'imageio==2.8.0',
        'omero-py==5.6.2',
        'zeroc-ice==3.6.5',
        'logfile_parser@git+https://git.ecdf.ed.ac.uk/jpietsch/logfile_parser@master'
    ]
)

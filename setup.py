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
        'pandas',
        'scikit-image>=0.16.2',
        'opencv-python',
        'requests_toolbelt',
        'h5py==2.10',
        'tables',
        'imageio==2.8.0',
        'omero-py>=5.6.2',
        'zeroc-ice==3.6.5',
        'baby@git+ssh://git@git.ecdf.ed.ac.uk/jpietsch/baby@dev',
        'logfile_parser@git+ssh://git@git.ecdf.ed.ac.uk/swain-lab/python-pipeline/logfile_parser.git',
        ],
)

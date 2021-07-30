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
        'pandas',
        'baby@git+https://git.ecdf.ed.ac.uk/jpietsch/baby@dev',
        'tensorflow >=1.14'
        'scikit-image<=0.16.2',
        'numpy',
        'tqdm',
        'opencv-python',
        'requests_toolbelt',
        'h5py',
        'imageio==2.8.0',
        'omero-py>=5.6.2',
        'zeroc-ice==3.6.5',
        'logfile_parser@git+https://git.ecdf.ed.ac.uk/jpietsch/logfile_parser@master'
        ]
)

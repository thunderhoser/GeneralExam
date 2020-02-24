"""Setup file for GeneralExam."""

from setuptools import setup

PACKAGE_NAMES = [
    'generalexam', 'generalexam.ge_io', 'generalexam.ge_utils',
    'generalexam.machine_learning', 'generalexam.evaluation',
    'generalexam.plotting', 'generalexam.scripts', 'generalexam.climo_paper'
]

SHORT_DESCRIPTION = 'General-exam stuff'

LONG_DESCRIPTION = (
    'Code for Ryan Lagerquist''s Ph.D. qualifying exam at University of '
    'Oklahoma, School of Meteorology, 2018.')

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7'
]

# You also need to install the following packages, which are not available in
# pip.  They can both be installed by "git clone" and "python setup.py install",
# the normal way one installs a GitHub package.
#
# https://github.com/matplotlib/basemap
# https://github.com/sharppy/SHARPpy
# https://github.com/thunderhoser/GewitterGefahr

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'astar',
]

if __name__ == '__main__':
    setup(name='GeneralExam',
          version='0.1',
          description=SHORT_DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license='MIT',
          author='Ryan Lagerquist',
          author_email='ryan.lagerquist@ou.edu',
          url='https://github.com/thunderhoser/GeneralExam',
          packages=PACKAGE_NAMES,
          scripts=[],
          classifiers=CLASSIFIERS,
          include_package_data=True,
          zip_safe=False,
          install_requires=PACKAGE_REQUIREMENTS)

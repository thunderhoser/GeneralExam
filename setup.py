"""Setup file for GeneralExam."""

from setuptools import setup

PACKAGE_NAMES = ['generalexam', 'generalexam.ge_io']
SHORT_DESCRIPTION = 'General-exam stuff'
LONG_DESCRIPTION = 'Code for 2018 Ph.D. general exam'

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7']

PACKAGE_REQUIREMENTS = [
    'descartes', 'geopy', 'netCDF4', 'pyproj', 'scipy', 'sharppy', 'skewt',
    'scikit-learn', 'matplotlib', 'numpy', 'pandas', 'shapely', 'scikit-image']

if __name__ == '__main__':
    setup(name='GeneralExam', version='0.1', description=SHORT_DESCRIPTION,
          author='Ryan Lagerquist', author_email='ryan.lagerquist@ou.edu',
          long_description=LONG_DESCRIPTION, license='MIT',
          url='https://github.com/thunderhoser/GeneralExam',
          packages=PACKAGE_NAMES, scripts=[],
          classifiers=CLASSIFIERS, include_package_data=True, zip_safe=False,
          install_requires=PACKAGE_REQUIREMENTS)

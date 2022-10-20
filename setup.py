from distutils.core import setup, Extension
import os
import glob
#from lib import __version__

setup(name='bfptc',
      version='0.2',
      description='Tools to analyze CCD flats in order to assemble photon transfer and covariance curves',
      author='P. Astier',
      author_email='pierre.astier@in2p3.fr',

      packages=['bfptc'],
      package_dir = {'bfptc': 'py'},
      scripts=glob.glob('tools/*.py')
)

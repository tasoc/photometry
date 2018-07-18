# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:53:19 2018

.. author: Rasmus Handberg <rasmush@phys.au.dk>
"""

from setuptools import setup

with open("README.rst", 'r') as f:
	long_description = f.read()

setup(
	name = 'tessphot',
	version = '0.3.0',
	description = 'A useful module',
	license = 'GPLv3',
	long_description = long_description,
	long_description_content_type = 'text/x-rst',
	author = 'Rasmus Handberg',
	author_email = 'rasmush@phys.au.dk',
	url = "https://tasoc.dk/code",
	packages = ['tessphot'], # same as name
	package_dir = {'tessphot': 'photometry'},
	install_requires = [ # external packages as dependencies
		'numpy',
		'scipy >= 0.16',
		'astropy >= 2.0, < 3.0',
		'photutils >= 0.4',
		'Bottleneck >= 1.2',
		'h5py',
		'enum-compat',
		'halophot'
	],
	classifiers = (
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 2.7',
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Operating System :: OS Independent',
		'Topic :: Scientific/Engineering :: Astronomy'
	)
)

#,
#	scripts = [
#		'scripts/tessphot-prepare',
#		'scripts/tessphot-catalog',
#		'scripts/tessphot-todo',
#	]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note on how test-data was created.

Warning: This will mess with the files in the test-directory!
"""

import os
import sys
import tempfile
from astropy.io import fits
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.utilities import download_file

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	INPUT_DIR = os.path.abspath(os.path.join('..', 'tests', 'input', 'images'))

	downlist = [
		{'path': 'hlsp_tess-data-alerts_tess_phot_00025155310-s01_tess_v1_tp.fits.gz'},
		{'path': 'tess2018206045859-s0001-0000000260795451-0120-s_tp.fits.gz'},
		{'path': 'tess2018206045859-s0001-0000000267211065-0120-s_tp.fits.gz', 'numpixels': 5000},
		{'path': 'tess2020186164531-s0027-0000000025155310-0189-a_fast-tp.fits.gz'},
		{'path': 'tess2020186164531-s0027-0000000025155310-0189-s_tp.fits.gz'},
	]

	# Add the URL for the files:
	urlbase = 'https://tasoc.dk/pipeline/photometry_test_images/'
	for d in downlist:
		d['url'] = urlbase + d['path']

	with tempfile.TemporaryDirectory() as tmpdir:
		for d in downlist:
			print(d)
			fpath = os.path.join(INPUT_DIR, d['path'])
			tmppath = os.path.join(tmpdir, d['path'])

			if os.path.exists(fpath):
				os.remove(fpath)

			# TODO: Use download_parallel instead?
			download_file(d['url'], tmppath, showprogress=True)

			# Optionally shorten the files to save some space:
			numpixels = d.get('numpixels', 1000)
			if numpixels is not None:
				with fits.open(tmppath, mode='readonly') as hdu:
					hdu['PIXELS'].data = hdu['PIXELS'].data[:numpixels]
					hdu.writeto(fpath, checksum=True)

			else:
				os.rename(tmppath, fpath)

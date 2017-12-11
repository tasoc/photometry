#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
from astropy.io import fits
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.backgrounds import fit_background

def test_background():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input', 'images')
	fname = glob.glob(os.path.join(INPUT_DIR, '*.fits.gz'))[0]

	hdr = fits.getheader(fname, ext=0)
	img_shape = (hdr['NAXIS1'], hdr['NAXIS2'])

	bck, mask = fit_background(fname)

	assert(bck.shape == img_shape)
	assert(mask.shape == img_shape)

if __name__ == '__main__':
	test_background()

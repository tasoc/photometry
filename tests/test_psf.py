#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os.path
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.psf import PSF

def test_psf():

	stamp = (50, 60, 120, 140)

	for camera in (1,2,3,4):
		for ccd in (1,2,3,4):

			psf = PSF(camera, ccd, stamp)

			stars = np.array([
				[psf.ref_row-psf.stamp[0], psf.ref_column-psf.stamp[2], 1],
			])

			img = psf.integrate_to_image(stars)

			assert np.unravel_index(np.argmax(img), img.shape) == (int(stars[0,0]), int(stars[0,1])), "Maximum not correct place"
			assert psf.camera == camera, "CAMERA not set"
			assert psf.ccd == ccd, "CCD not set"
			assert img.shape == (stamp[1]-stamp[0], stamp[3]-stamp[2]), "not the right size"
			assert img.shape == psf.shape, "Not the right size either"

			#psf.plot()

if __name__ == '__main__':
	test_psf()

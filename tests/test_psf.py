#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of PSF object.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os.path
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.psf import PSF
from photometry.plots import plt

#--------------------------------------------------------------------------------------------------
def test_psf(keep_figures=False):

	stamp = (50, 60, 120, 140)

	for sector in (2, 5):
		for camera in (1,2,3,4):
			for ccd in (1,2,3,4):

				psf = PSF(sector, camera, ccd, stamp)

				stars = np.array([
					[psf.ref_row-psf.stamp[0], psf.ref_column-psf.stamp[2], 1],
				])

				img = psf.integrate_to_image(stars)

				print(psf.PSFfile)

				assert np.unravel_index(np.argmax(img), img.shape) == (int(stars[0,0]), int(stars[0,1])), "Maximum not correct place"
				assert psf.sector == sector, "SECTOR not set"
				assert psf.camera == camera, "CAMERA not set"
				assert psf.ccd == ccd, "CCD not set"
				assert img.shape == (stamp[1]-stamp[0], stamp[3]-stamp[2]), "not the right size"
				assert img.shape == psf.shape, "Not the right size either"

				fig = psf.plot()

				if not keep_figures:
					plt.close(fig)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	test_psf()

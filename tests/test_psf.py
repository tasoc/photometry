#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of PSF object.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import conftest # noqa: F401
from photometry.psf import PSF
from photometry.plots import plots_noninteractive, plt

#--------------------------------------------------------------------------------------------------
def test_psf_invalid_input():

	stamp = (50, 60, 120, 140)

	with pytest.raises(ValueError) as e:
		PSF(0, 1, 1, stamp)
	assert str(e.value) == 'Sector number must be greater than zero'

	with pytest.raises(ValueError) as e:
		PSF(1, 5, 1, stamp)
	assert str(e.value) == 'Camera must be 1, 2, 3 or 4.'

	with pytest.raises(ValueError) as e:
		PSF(12, 1, 5, stamp)
	assert str(e.value) == 'CCD must be 1, 2, 3 or 4.'

	with pytest.raises(ValueError) as e:
		PSF(12, 1, 1, [1, 2, 3, 4, 5])
	assert str(e.value) == 'Incorrect stamp provided.'

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('sector', (2,5))
@pytest.mark.parametrize('camera', (1,2,3,4))
@pytest.mark.parametrize('ccd', (1,2,3,4))
def test_psf(sector, camera, ccd):

	close_figures = True

	if close_figures:
		plots_noninteractive()

	stamp = (50, 60, 120, 140)

	psf = PSF(sector, camera, ccd, stamp)
	print(psf.PSFfile)

	stars = np.array([
		[psf.ref_row-psf.stamp[0], psf.ref_column-psf.stamp[2], 1],
	])

	img = psf.integrate_to_image(stars)

	assert np.unravel_index(np.argmax(img), img.shape) == (int(stars[0,0]), int(stars[0,1])), "Maximum not correct place"
	assert psf.sector == sector, "SECTOR not set"
	assert psf.camera == camera, "CAMERA not set"
	assert psf.ccd == ccd, "CCD not set"
	assert img.shape == (stamp[1]-stamp[0], stamp[3]-stamp[2]), "not the right size"
	assert img.shape == psf.shape, "Not the right size either"

	fig = psf.plot()

	if close_figures:
		plt.close(fig)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])

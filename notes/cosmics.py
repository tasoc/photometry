#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astroscrappy import detect_cosmics
from plots import plt, plot_image

#--------------------------------------------------------------------------------------------------
def cosmics(img, bck=0):

	mask = ~np.isfinite(img)

	crmask, cleanarr = detect_cosmics(img + bck,
		inmask=mask,
		sigclip=4.5,
		sigfrac=0.3,
		objlim=5.0,
		niter=4,
		pssl=0,
		gain=5.5,
		readnoise=6.5,
		satlevel=1e7,
		fsmode='convolve',
		psffwhm=3.5,
		verbose=False)

	return crmask

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	from astropy.io import fits
	from copy import deepcopy
	plt.switch_backend('Qt5Agg')


	with fits.open('tess-s0003-00129764561-0000-x_tpr.fits.gz', mode='readonly', memmap=True) as tpf:
		images = tpf['PIXELS'].data['RAW_CNTS']

		#img = 0
		#tmpimg = np.zeros((10, images[0].shape[0], images[0].shape[1]))

		for k in range(len(images)):

			#tmpimg = np.roll(tmpimg, 1, axis=0)
			#tmpimg[0,:,:] = img
			#previmg = np.nanmedian(tmpimg, axis=0)

			img = np.asarray(images[k], dtype='float64')

			crmask = cosmics(img)

			#if np.any(crmask):
			plt.figure()
			plt.subplot(121)
			plot_image(img, xlabel=None, ylabel=None, make_cbar=True)
			plt.subplot(122)
			plot_image(crmask, xlabel=None, ylabel=None, make_cbar=True)
			plt.show()

	print("Done")
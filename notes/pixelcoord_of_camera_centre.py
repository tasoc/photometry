#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, unicode_literals
import six
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
from astropy.wcs import WCS
import itertools
import os.path
from tqdm import trange

if __name__ == '__main__':

	rootdir = r'/aadc/tasoc/archive/S01_DR01'


	xycendict = {}
	for camera, ccd in itertools.product((1,2,3,4), (1,2,3,4)):

		if camera == 1:
			camera_centre = [324.566998914166, -33.172999301379]
		elif camera == 2:
			camera_centre = [338.57656612933, -55.0789269350771]
		elif camera == 3:
			camera_centre = [19.4927827153412, -71.9781542628999]
		elif camera == 4:
			camera_centre = [90.0042379538484, -66.5647239768875]


		with h5py.File(os.path.join(rootdir, 'sector001_camera{camera:d}_ccd{ccd:d}.hdf5'.format(camera=camera, ccd=ccd)), 'r') as hdf:

			N = len(hdf['images'])
			a = np.full(N, np.NaN)
			b = np.full(N, np.NaN)
			cno = np.arange(0, N, 1)

			for k in trange(N):
				if hdf['quality'][k] == 0:
					hdr_string = hdf['wcs']['%04d' % k][0]
					if not isinstance(hdr_string, six.string_types): hdr_string = hdr_string.decode("utf-8") # For Python 3
					wcs = WCS(header=fits.Header().fromstring(hdr_string), relax=True)

					xycen = wcs.all_world2pix(np.atleast_2d(camera_centre), 0, ra_dec_order=True)

					a[k] = xycen[0][0]
					b[k] = xycen[0][1]

			am = np.nanmedian(a)
			bm = np.nanmedian(b)

			plt.figure()
			plt.scatter(cno, a)
			plt.axhline(am)

			plt.figure()
			plt.scatter(cno, b)
			plt.axhline(bm)

			plt.show()

		# Save the
		xycendict[(camera, ccd)] = np.array([am, bm])


	print("xycen = {")
	for key, value in xycendict.items():
		print("\t(%d, %d): [%f, %f],"%(
			key[0],
			key[1],
			value[0],
			value[1]
		))
	print("}.get((camera, ccd))")
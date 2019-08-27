#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code to generate plot for Photometry Paper.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import h5py
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.plots import plt, plot_image
from photometry.quality import PixelQualityFlags
from matplotlib.colors import ListedColormap

if __name__ == '__main__':

	k = 98
	vmin = 0
	vmax = 500

	vmin2 = vmin
	vmax2 = vmax

	dset_name = '%04d' % k

	with h5py.File('sector001_camera1_ccd2.hdf5', 'r') as hdf:
		flux0 = np.asarray(hdf['images/' + dset_name])
		bkg = np.asarray(hdf['backgrounds/' + dset_name])
		#flags = np.asarray(hdf['pixel_flags/' + dset_name])
		flags = np.zeros_like(flux0, dtype='int32')
		flags[512:1024,512:1024] = 128

	flags = np.where(flags == 0, 0, np.log2(flags)+1)

	#flags = (flags & PixelQualityFlags.BackgroundShenanigans != 0)

	white = np.array([1, 1, 1, 1])

	viridis = plt.get_cmap('viridis')
	newcolors = viridis(np.linspace(0, 1, int(np.max(flags))))
	newcolors[:1, :] = white
	newcmp = ListedColormap(newcolors)

	fig, ax = plt.subplots(1, 4, figsize=(20, 6))

	plot_image(flux0+bkg, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	plot_image(bkg, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	plot_image(flux0, ax=ax[2], scale='sqrt', vmin=vmin2, vmax=vmax2, title='Background subtracted', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	plot_image(flags, ax=ax[3], scale='linear', vmin=-0.5, vmax=np.max(flags)-0.5, title='Pixel Flags', xlabel=None, ylabel=None, cmap=newcmp, make_cbar=True, clabel='Bit')

	for a in ax:
		a.set_xticks([])
		a.set_yticks([])

	fig.set_tight_layout('tight')

	fig.savefig('sector001_camera1_ccd2.png', bbox_inches='tight')
	#plt.close(fig)
	plt.show()
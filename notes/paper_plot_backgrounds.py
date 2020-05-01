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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.plots import plt, plot_image, matplotlib
from photometry.quality import PixelQualityFlags
from matplotlib.colors import ListedColormap
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = '14'
matplotlib.rcParams['axes.titlesize'] = '18'
matplotlib.rcParams['axes.labelsize'] = '16'
plt.rc('text', usetex=True)

from rasterize_and_save import rasterize_and_save

if __name__ == '__main__':

	# Which timestamp to show:
	k = 98

	# Load the data from the HDF5 file:
	with h5py.File('sector001_camera1_ccd2.hdf5', 'r') as hdf:
		dset_name = '%04d' % k
		flux0 = np.asarray(hdf['images/' + dset_name])
		bkg = np.asarray(hdf['backgrounds/' + dset_name])
		img = np.asarray(hdf['pixel_flags/' + dset_name])
		#img = np.zeros_like(flux0, dtype='int32')
		#img[512:1024,512:1024] = 128

		flags = np.zeros_like(img, dtype='uint8')
		flags[img & PixelQualityFlags.NotUsedForBackground != 0] = 1
		flags[img & PixelQualityFlags.ManualExclude != 0] = 2
		flags[img & PixelQualityFlags.BackgroundShenanigans != 0] = 3

		vmin = hdf['backgrounds'].attrs.get('movie_vmin')
		vmax = hdf['backgrounds'].attrs.get('movie_vmax')
		vmin2 = hdf['images'].attrs.get('movie_vmin')
		vmax2 = hdf['images'].attrs.get('movie_vmax')

	print(vmin, vmax)
	print(vmin2, vmax2)

	# Colormap for images:
	cmap = plt.cm.viridis

	# Colormap for Flags:
	viridis = plt.get_cmap('Dark2')
	newcolors = viridis(np.linspace(0, 1, 4))
	newcolors[:1, :] = np.array([1, 1, 1, 1])
	cmap_flags = ListedColormap(newcolors)

	# Create figures:
	fig, ax = plt.subplots(1, 4, figsize=(20, 6.2))
	img1 = plot_image(flux0+bkg, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image', xlabel=None, ylabel=None, cmap=cmap, make_cbar=True)
	img2 = plot_image(bkg, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', xlabel=None, ylabel=None, cmap=cmap, make_cbar=True)
	img3 = plot_image(flux0, ax=ax[2], scale='sqrt', vmin=vmin2, vmax=vmax2, title='Background subtracted', xlabel=None, ylabel=None, cmap=cmap, make_cbar=True)
	img4 = plot_image(flags, ax=ax[3], scale='linear', vmin=-0.5, vmax=3.5, title='Pixel Flags', xlabel=None, ylabel=None, cmap=cmap_flags, make_cbar=True, clabel='Flags', cbar_ticks=[0,1,2,3], cbar_ticklabels=['None','Not used','Man. Excl.','Shenan'])

	# Remove axes ticks:
	for a in ax:
		a.set_xticks([])
		a.set_yticks([])

	fig.set_tight_layout('tight')

	# Save figure to file:
	#rasterize_and_save('sector001_camera1_ccd2.pdf', [img1, img2, img3, img4], fig=fig, dpi=150, bbox_inches='tight')
	fig.savefig('sector001_camera1_ccd2.png', bbox_inches='tight', dpi=150)
	plt.close(fig)
	#plt.show()

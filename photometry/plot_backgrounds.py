#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, unicode_literals
import numpy as np
import h5py
from plots import plt, plot_image
from matplotlib import animation
from tqdm import tqdm, trange
from scipy.ndimage.filters import median_filter
from scipy.ndimage import generic_filter
from utilities import move_median_central
from timeit import default_timer
from bottleneck import nanmedian, replace
import cv2
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
from quality import TESSQualityFlags


def _animate(k, imgs, ax, hdf_file, SumImage, limit, fname):

	with h5py.File(hdf_file, 'r', libver='latest') as hdf:
		dset_name = '%04d' % k
		images = hdf['images']
		backgrounds = hdf['backgrounds']

		flux0 = np.asarray(images[dset_name])
		bkg = np.asarray(backgrounds[dset_name])

		if os.path.exists(fname):
			img = np.load(fname)
		else:

			img = flux0 - SumImage

			#tic = default_timer()
			#img = np.clip(np.abs(img), 0, 40)
			#img -= np.min(img)
			#img = np.asarray(np.round(256*img/np.max(img)), dtype='uint8')
			#img = cv2.resize(img, img.shape)
			#img = cv2.medianBlur(img, 15)
			#toc = default_timer()
			#print(toc-tic)

			#tic = default_timer()
			img = median_filter(img, size=15)
			#toc = default_timer()

			np.save(fname, img)


		img[(-limit < img) & (img < limit)] = 0

		# Plot background:
		imgs[0].set_data(flux0 + bkg)
		#ax[0].set_title('Original Image - %d' % k)
		imgs[1].set_data(bkg)
		imgs[2].set_data(flux0)
		imgs[3].set_data(img)

	return imgs


if __name__ == '__main__':

	hdf_file = r'E:\tess_data\S01_DR01\sector001_camera1_ccd2.hdf5'

	plt.close('all')

	with h5py.File(hdf_file, 'r', libver='latest') as hdf:
		numfiles = len(hdf['images'])
		images = hdf['images']
		backgrounds = hdf['backgrounds']
		SumImage = np.asarray(hdf['sumimage'])

		dummy_img = np.asarray(images['0000'])
		dummy_bkg = np.asarray(backgrounds['0000'])


		vmin = 95.87338256835938
		vmax = 1345.383056640625

		#vmax = -np.inf
		#vmin = np.inf
		#for k in trange(numfiles):
		#	vmin1, vmax1 = np.nanpercentile(backgrounds['%04d' % k], [0, 100])
		#	vmin = min(vmin, vmin1)
		#	vmax = max(vmax, vmax1)
		#vmax *= 1.5
		#vmin /= 2.0
		print(vmin, vmax)

	limit = 30

	#viridis = cm.get_cmap('seismic', 400)
	#newcolors = viridis(np.linspace(0, 1, 400))
	#newcolors = np.zeros((400, 4))
	#white = np.array([1, 1, 1, 1])
	#newcolors[:(200-limit), :] = viridis(np.linspace(0, 0.5, 200-limit))
	#newcolors[(200-limit):(200+lim), :] = white
	#newcolors[(200+limit):, :] = viridis(np.linspace(0.5, 1, 200-limit))
	#newcmp = ListedColormap(newcolors)


	fig, ax = plt.subplots(1, 4, figsize=(20, 6))

	#dummy = np.ones((2048, 2048))
	imgs = [0,0,0,0]
	imgs[0] = plot_image(dummy_bkg, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	imgs[1] = plot_image(dummy_bkg, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	imgs[2] = plot_image(dummy_img, ax=ax[2], scale='sqrt', vmin=0, title='Background subtracted', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	imgs[3] = plot_image(dummy_img, ax=ax[3], scale='linear', vmin=-200, vmax=200, title='Background Shenanigans', xlabel=None, ylabel=None, cmap=plt.cm.seismic, make_cbar=True)

	for a in ax:
		a.set_xticks([])
		a.set_yticks([])

	plt.tight_layout()

	#fig3 = plt.figure()
	#axtmp = fig3.add_subplot(111)
	#plt.show()

	for sheiters in range(2):
		for k in trange(numfiles):
			if not os.path.exists('slow3/%02d-%04d.png' % (sheiters, k)):
				fname = 'slow3/%02d-%04d.npy' % (sheiters, k)
				_animate(k, imgs, ax, hdf_file, SumImage, limit, fname)
				fig.savefig('slow3/%02d-%04d.png' % (sheiters, k), bbox_inches='tight')

		if os.path.exists('slow3/%02d-sumimage2.npy' % sheiters):
			SumImage2 = np.load('slow3/%02d-sumimage2.npy' % sheiters)
		else:
			with h5py.File(hdf_file, 'r', libver='latest') as hdf:
				numfiles = len(hdf['images'])
				images = hdf['images']
				backgrounds = hdf['backgrounds']
				quality = np.asarray(hdf['quality'])

				SumImage2 = np.zeros_like(SumImage)
				Nimg = np.zeros_like(SumImage2, dtype='int32')
				for k in trange(numfiles):
					if TESSQualityFlags.filter(quality[k]):
						dset_name = '%04d' % k

						flux0 = np.asarray(images[dset_name])

						img = np.load(('slow3/%02d-'%sheiters) + dset_name + '.npy')
						bkgshe = (np.abs(img) > limit)

						#plot_image(bkgshe, scale='linear', vmin=0, vmax=1, ax=axtmp, cmap=plt.cm.seismic)
						#plt.pause(0.05)

						flux0[bkgshe] = np.nan

						#plot_image(flux0, scale='sqrt', ax=axtmp)
						#plt.pause(0.05)

						Nimg += np.isfinite(flux0)
						replace(flux0, np.nan, 0)
						SumImage2 += flux0

				SumImage2 /= Nimg
			np.save('slow3/%02d-sumimage2.npy' % sheiters, SumImage2)

		SumImage = np.copy(SumImage2)

		fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
		plot_image(SumImage, scale='sqrt', ax=ax2[0])
		plot_image(SumImage2, scale='sqrt', ax=ax2[1])
		fig2.savefig('slow3/%02d-sumimages.png' % sheiters)

	#ani = animation.FuncAnimation(fig, _animate, trange(numfiles), fargs=(imgs, ax, hdf_file, SumImage), repeat=False, blit=False)
	#ani.save('test.gif', writer='pillow')
	plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import h5py
import numpy as np
from tqdm import trange
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.plots import plt, plots_interactive, plot_image
from matplotlib import animation

if __name__ == '__main__':
	plots_interactive()

	xycen = [-5.653058, 2098.018608]
	xx, yy = np.meshgrid(
		np.arange(44, 2048+44, 1),
		np.arange(0, 2048, 1)
	)
	r = np.sqrt((xx - xycen[0])**2 + (yy - xycen[1])**2)

	rootdir = r'G:\tess_data\S01_DR01'

	with h5py.File(os.path.join(rootdir, 'sector001_camera1_ccd2.hdf5')) as hdf, h5py.File(os.path.join(rootdir, 'sector001_camera1_ccd2-nocorner.hdf5')) as hdf_nocorner:

		N = len(hdf['backgrounds'])
		d = np.empty([N, 6], dtype='float64')

		stdimg = np.empty(N, dtype='float64')
		stdimg_nocorner = np.empty(N, dtype='float64')

		fig, ax = plt.subplots()
		im = plot_image(r, ax=ax, cbar='right', scale='linear', cmap='seismic', vmin=-1.5, vmax=1.5, clabel='Relative difference')

		WriterClass = animation.writers['ffmpeg']
		writer = WriterClass(fps=15, codec='h264', bitrate=-1)
		with writer.saving(fig, 'output.mp4', 100):
			for k in trange(N):
				dset = f"{k:04d}"

				img = np.array(hdf['backgrounds/' + dset])
				img_nocorner = np.array(hdf_nocorner['backgrounds/' + dset])

				img2 = img.copy(); img2[r < 2400] = np.NaN
				img_nocorner2 = img_nocorner.copy(); img_nocorner2[r < 2400] = np.NaN
				stdimg[k] = np.nansum(img2)
				stdimg_nocorner[k] = np.nansum(img_nocorner2)

				diff = img/img_nocorner - 1

				im.set_data(diff)

				diff[r < 2400] = np.NaN
				d[k, :] = np.nanpercentile(diff, [5, 20, 50, 80, 95, 99])
				#print(d[k, :])

				ax.set_title(f"{dset:s} - {d[k,0]:f} - {d[k,2]:f}")

				writer.grab_frame()

		plt.close(fig)

	dsets = np.arange(N)

	fig, ax = plt.subplots()
	ax.plot(dsets, d[:,0], label='5')
	ax.plot(dsets, d[:,1], label='20')
	ax.plot(dsets, d[:,2], label='50')
	ax.plot(dsets, d[:,3], label='80')
	ax.plot(dsets, d[:,4], label='95')
	ax.plot(dsets, d[:,5], label='99')
	ax.set_ylabel('Rel. Difference')
	ax.set_xlabel('Dataset')
	plt.legend()

	fig, ax = plt.subplots()
	ax.plot(dsets, stdimg)
	ax.plot(dsets, stdimg_nocorner, label='No corner')
	ax.set_xlabel('Dataset')
	plt.legend()

	fig, ax = plt.subplots()
	ax.plot(dsets, stdimg / stdimg_nocorner)
	ax.set_xlabel('Dataset')

	plt.show()

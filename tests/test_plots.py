#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of plots

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os.path
import numpy as np
#from matplotlib.testing.decorators import image_comparison
import scipy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.plots import plt, plot_image

#@image_comparison(baseline_images=['plot_image'], extensions=['png'])
def test_plot_image():

	mu = [3.5, 3]
	x, y = np.mgrid[0:10, 0:10]
	pos = np.dstack((x, y))
	var = scipy.stats.multivariate_normal(mean=mu, cov=[[1,0],[0,1]])
	gauss = var.pdf(pos)

	fig = plt.figure(figsize=(12,6))
	ax1 = fig.add_subplot(131)
	plot_image(gauss, ax=ax1, scale='linear', title='Linear')
	ax1.plot(mu[1], mu[0], 'r+')
	ax2 = fig.add_subplot(132)
	plot_image(gauss, ax=ax2, scale='sqrt', title='Sqrt')
	ax2.plot(mu[1], mu[0], 'r+')
	ax3 = fig.add_subplot(133)
	plot_image(gauss, ax=ax3, scale='log', title='Log')
	ax3.plot(mu[1], mu[0], 'r+')

#@image_comparison(baseline_images=['plot_image_grid1', 'plot_image_grid2'], extensions=['png'])
def test_plot_image_grid():

	img = np.zeros((5,7))
	img[:,0] = 1
	img[0,:] = 1
	img[:,-1] = 1
	img[-1,:] = 1

	fig = plt.figure()
	plot_image(img, scale='linear')
	plt.plot(0.5, 0.5, 'r+')
	plt.plot(5.5, 3.5, 'g+')
	plt.grid(True)

	fig = plt.figure()
	plot_image(img, scale='linear', offset_axes=(3,2))
	plt.plot(3.5, 2.5, 'r+')
	plt.plot(8.5, 5.5, 'g+')
	plt.grid(True)


if __name__ == '__main__':
	plt.close('all')
	test_plot_image()
	test_plot_image_grid()
	plt.show()

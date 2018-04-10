#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of plots

>>> pytest --mpl

>>> pytest --mpl-generate-path=tests/baseline_images

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os.path
import numpy as np
import scipy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.plots import plt, plot_image
import pytest

kwargs = {'baseline_dir': 'baseline_images', 'savefig_kwargs': {'bbox_inches': 'tight'}}

@pytest.mark.mpl_image_compare(**kwargs)
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

	return fig

@pytest.mark.mpl_image_compare(**kwargs)
def test_plot_image_grid():

	img = np.zeros((5,7))
	img[:,0] = 1
	img[0,:] = 1
	img[:,-1] = 1
	img[-1,:] = 1

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plot_image(img, ax=ax, scale='linear')
	ax.plot(0.5, 0.5, 'r+')
	ax.plot(5.5, 3.5, 'g+')
	ax.grid(True)
	return fig

@pytest.mark.mpl_image_compare(**kwargs)
def test_plot_image_grid_offset():

	img = np.zeros((5,7))
	img[:,0] = 1
	img[0,:] = 1
	img[:,-1] = 1
	img[-1,:] = 1

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plot_image(img, ax=ax, scale='linear', offset_axes=(3,2))
	ax.plot(3.5, 2.5, 'r+')
	ax.plot(8.5, 5.5, 'g+')
	ax.grid(True)
	return fig

if __name__ == '__main__':
	plt.close('all')
	test_plot_image()
	test_plot_image_grid()
	test_plot_image_grid_offset()
	plt.show()

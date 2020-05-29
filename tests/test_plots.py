#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of plots

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import numpy as np
from scipy.stats import multivariate_normal
import conftest # noqa: F401
from photometry.plots import plt, plot_image, plot_image_fit_residuals

kwargs = {
	'baseline_dir': os.path.join(os.path.abspath(os.path.dirname(__file__)), 'correct_plots'),
	'tolerance': 30
}

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpl_image_compare(**kwargs)
def test_plot_image():

	mu = [3.5, 3]
	x, y = np.mgrid[0:15, 0:15]
	pos = np.dstack((x, y))
	var = multivariate_normal(mean=mu, cov=[[1,0],[0,1]])
	gauss = 5 * var.pdf(pos) - 0.05 # Make sure it has some negative values as well
	gauss[8,8] = np.NaN
	gauss[4,4] = -0.2

	scales = ['linear', 'sqrt', 'log', 'asinh', 'histeq', 'sinh', 'squared']

	fig, axes = plt.subplots(2, 4, figsize=(14, 8))
	axes = axes.flatten()
	for k, scale in enumerate(scales):
		ax = axes[k]
		plot_image(gauss, ax=ax, scale=scale, title=scale, cbar='right')
		ax.plot(mu[1], mu[0], 'r+')

	# In the final plot:
	plot_image(gauss, ax=axes[-1], scale='log', title='log - Reds', cmap='Reds', cbar='right')

	fig.tight_layout()

	return fig

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpl_image_compare(**kwargs)
def test_plot_image_invalid():

	mu = [3.5, 3]
	x, y = np.mgrid[0:10, 0:10]
	pos = np.dstack((x, y))
	var = multivariate_normal(mean=mu, cov=[[1,0],[0,1]])
	gauss = var.pdf(pos)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

	# Run with invalid scale:
	with pytest.raises(ValueError):
		plot_image(gauss, ax=ax1, scale='invalid-scale')

	# Plot with single NaN:
	gauss[1,1] = np.NaN
	plot_image(gauss, ax=ax1, scale='log')

	# Run with all-NaN image:
	gauss[:, :] = np.NaN
	plot_image(gauss, ax=ax2, cbar='right')
	return fig

#--------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------
def test_plot_image_data_change():
	"""Test that the plotting function does not change input data"""

	# Construct random image:
	np.random.seed(42)
	img = np.random.randn(15, 10)
	img[0,0] = -1.0 # Make 100% sure there is a negative point

	# Save the original image for comparison:
	img_before = np.copy(img)

	# Make a couple of plots trying out the different settings:
	fig = plt.figure()
	ax1 = fig.add_subplot(131)
	plot_image(img, ax=ax1, scale='linear')
	np.testing.assert_allclose(img, img_before)

	ax2 = fig.add_subplot(132)
	plot_image(img, ax=ax2, scale='sqrt')
	np.testing.assert_allclose(img, img_before)

	ax3 = fig.add_subplot(133)
	plot_image(img, ax=ax3, scale='log')
	np.testing.assert_allclose(img, img_before)

	fig = plt.figure()
	plot_image_fit_residuals(fig, img, img, img)
	np.testing.assert_allclose(img, img_before)

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpl_image_compare(**kwargs)
def test_plot_cbar_and_nans():

	# Construct image:
	np.random.seed(42)
	img = np.random.rand(10, 10)
	img[2:8, 2:8] = np.NaN

	fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 6))
	plot_image(img, ax=ax1, scale='linear', vmin=0.4, cbar='left')
	plot_image(img, ax=ax2, scale='sqrt', vmin=0.4, cbar='bottom')
	plot_image(img, ax=ax3, scale='log', vmin=0.4, cbar='right')
	plot_image(img, ax=ax4, scale='asinh', vmin=0.4, cbar='top')
	return fig

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	print("To generate new correct images:")
	print('pytest --mpl-generate-path="' + kwargs['baseline_dir'] + '" "' + __file__ + '"')

	plt.switch_backend('Qt5Agg')
	pytest.main([__file__])
	plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting utilities.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import (PercentileInterval, ImageNormalize,
								   SqrtStretch, LogStretch, LinearStretch)

def plot_image(image, scale='log', origin='lower', xlabel='Pixel Column Number',
			   ylabel='Pixel Row Number', make_cbar=True, clabel='Flux ($e^{-}s^{-1}$)', 
			   title=None, norm=None,**kwargs):
	"""Utility function to plot a 2D image.

	Parameters:
		image (2d array): Image data.
		scale (str, optional): Scale used to stretch the colormap. Options: 'linear', 'sqrt', or 'log'.
		origin (str, optional): The origin of the coordinate system.
		xlabel (str, optional): Label for the x-axis.
		ylabel (str, optional): Label for the y-axis.
		make_cbar (boolean): ``False`` if plot without colorbar. Default is ``True``.
		clabel (str, optional): Label for the color bar.
		title (str or None, optional): Title for the plot.
		norm (astropy.visualization.mpl_normalize.ImageNormalize object): Output 
		from `astropy.visualization.ImageNormalize`. Default is ``None`` in which 
		case the normalization is derived from the image.
		kwargs (dict, optional): Keyword arguments to be passed to `matplotlib.pyplot.imshow`.
	"""

	if norm is None:
		vmin, vmax = PercentileInterval(95.).get_limits(image)
	
		if scale == 'linear':
			norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
		elif scale == 'sqrt':
			norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
		elif scale == 'log':
			norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
		else:
			raise ValueError("scale {} is not available.".format(scale))

	extent = (0, image.shape[0], 0, image.shape[1])
	
	im = plt.imshow(image, origin=origin, norm=norm, extent=extent, **kwargs)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)

	if make_cbar:
		cbar = plt.colorbar(norm=norm)
		cbar.set_label(clabel)

	return im


def plot_image_fit_residuals(fig, image, fit, residuals):
	"""
	Make a figure with three subplots showing the image, the fit and the 
	residuals. The image and the fit are shown with logarithmic scaling and a
	common colorbar. The residuals are shown with linear scaling and a seperate
	colorbar.
	
	Parameters:
		fig (fig object): Figure object in which to make the subplots.
		image (2D array): Image numpy array.
		fit (2D array): Fitted image numpy array.
		residuals (2D array): Fitted image subtracted from image numpy array.
		positions (list of arrays): List with the catalog and PSF fitted 
		centroid positions. Format is (row,col). Default is ``None`` which does
		not plot the positions.
	
	Returns:
		axes (list): List with Matplotlib subplot axes objects for each subplot.
	"""

	# Calculate common normalization for the first two subplots:
	vmin_image, vmax_image = PercentileInterval(95.).get_limits(image)
	vmin_fit, vmax_fit = PercentileInterval(95.).get_limits(fit)
	vmin = np.nanmin([vmin_image, vmin_fit])
	vmax = np.nanmax([vmax_image, vmax_fit])
	norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())

	# Add subplot with the image:
	ax1 = fig.add_subplot(131)
	im1 = plot_image(image, norm=norm, make_cbar=False)

	# Add subplot with the fit:
	ax2 = fig.add_subplot(132)
	plot_image(fit, norm=norm, make_cbar=False)

	# Make the common colorbar for image and fit subplots:
	cbar_ax12 = fig.add_axes([0.125, 0.2, 0.494, 0.03])
	fig.colorbar(im1, norm=norm, cax=cbar_ax12, orientation='horizontal')
	cbar_ax12.set_xticklabels(cbar_ax12.get_xticklabels(), rotation='vertical')

	# Calculate the normalization for the third subplot:
	vmin, vmax = PercentileInterval(95.).get_limits(residuals)
	norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

	# Add subplot with the residauls:
	ax3 = fig.add_subplot(133)
	im3 = plot_image(residuals, scale='linear', make_cbar=False)

	# Make the colorbar for the residuals subplot:
	cbar_ax3 = fig.add_axes([0.7, 0.2, 0.205, 0.03])
	fig.colorbar(im3, norm=norm, cax=cbar_ax3, orientation='horizontal')
	cbar_ax3.set_xticklabels(cbar_ax3.get_xticklabels(), rotation='vertical')

	# Add more space between subplots:
	plt.subplots_adjust(wspace=0.4, hspace=0.4)
	
	# Set titles:
	ax_list = [ax1, ax2, ax3]
	title_list = ['Image', 'PSF fit', 'Residuals']
	for ax, title in zip(ax_list, title_list):
		ax.set_title(title)

	return ax_list


def save_figure(output_folder, fig_name, ftype='png'):
	"""
	Write current figure to file. Creates directory to place it in if needed.
	
	Parameters:
		output_folder (string): Path to directory where to save figure.
		fig_name (string): Figure name.
		ftype (string): Figure file type. Default is ``'png'``.
	
	"""

	logger = logging.getLogger(__name__)
	logger.info('Saving figure'+fig_name+'.'+ftype+' to '+output_folder)

	# Make figure output directory if it doesn't exist:
	fig_output_folder = os.path.join(output_folder, 'figures')
	if not os.path.exists(fig_output_folder):
		os.makedirs(fig_output_folder)

	# Write current figure to file if it doesn't exist:
	file_dir = os.path.join(fig_output_folder, fig_name+'.'+ftype)
	if not os.path.exists(file_dir):
		plt.savefig(file_dir)
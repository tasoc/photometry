#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plotting utilities.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import matplotlib.pyplot as plt
from astropy.visualization import (PercentileInterval, ImageNormalize,
								   SqrtStretch, LogStretch, LinearStretch)

def plot_image(image, scale='log', origin='lower', xlabel='Pixel Column Number',
			   ylabel='Pixel Row Number', clabel='Flux ($e^{-}s^{-1}$)', title=None, **kwargs):
	"""Utility function to plot a 2D image.

	Parameters:
		image (2d array): Image data.
		scale (str, optional): Scale used to stretch the colormap. Options: 'linear', 'sqrt', or 'log'.
		origin (str, optional): The origin of the coordinate system.
		xlabel (str, optional): Label for the x-axis.
		ylabel (str, optional): Label for the y-axis.
		clabel (str, optional): Label for the color bar.
		title (str or None, optional): Title for the plot.
		kwargs (dict, optional): Keyword arguments to be passed to `matplotlib.pyplot.imshow`.
	"""

	vmin, vmax = PercentileInterval(95.).get_limits(image)

	if scale == 'linear':
		norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
	elif scale == 'sqrt':
		norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
	elif scale == 'log':
		norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
	else:
		raise ValueError("scale {} is not available.".format(scale))

	plt.imshow(image, origin=origin, norm=norm, **kwargs)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	cbar = plt.colorbar(norm=norm)
	cbar.set_label(clabel)

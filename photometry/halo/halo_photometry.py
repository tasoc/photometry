#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Halo Photometry.

.. codeauthor:: Tim White <white@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import logging
import os.path
import numpy as np
from ..plots import plt, save_figure, plot_image
from .. import BasePhotometry, STATUS
from ..utilities import mag2flux
import halophot
from halophot.halo_tools import do_lc
from astropy.table import Table

#------------------------------------------------------------------------------
class HaloPhotometry(BasePhotometry):
	"""Use halo photometry to observe very saturated stars.

	.. codeauthor:: Benjamin Pope <benjamin.pope@nyu.edu>
	.. codeauthor:: Tim White <white@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Here you could do other things that needs doing in the beginning
		# of the run on each target.


	def do_photometry(self):
		"""Performs 'halo' TV-min weighted-aperture photometry.

		Parameters
		----------
		aperture_mask : array-like, 'pipeline', or 'all'
			A boolean array describing the aperture such that `False` means
			that the pixel will be masked out.
			If the string 'all' is passed, all pixels will be used.
			The default behaviour is to use the Kepler pipeline mask.
		splits : tuple, (None, None) or (2152,2175) etc.
			A tuple including two times at which to split the light curve and run halo
			separately outside these splits.
		sub : int
			Do you want to subsample every nth pixel in your light curve? Not advised,
			but can come in handy for very large TPFs.
		order: int
			Run nth order TV - ie first order is L1 norm on first derivative,
			second order is L1 norm on second derivative, etc.
			This is part of the Pock generalized TV scheme, so that
			1st order gives you piecewise constant functions,
			2nd order gives you piecewise affine functions, etc.
			Currently implemented only up to 2nd order in numerical, 1st in analytic!
			We recommend first order very strongly.
		maxiter: int
			Number of iterations to optimize. 101 is default & usually sufficient.
		w_init: None or array-like.
			Initialize weights with a particular weight vector - useful if you have
			already run TV-min and want to update, but otherwise set to None
			and it will have default initialization.
		random_init: Boolean
			If False, and w_init is None, it will initialize with uniform weights; if True, it
			will initialize with random weights. False is usually better.
		thresh: float
			A float greater than 0. Pixels less than this fraction of the maximum
			flux at any pixel will be masked out - this is to deal with saturation.
			Because halo is usually intended for saturated stars, the default is 0.8,
			to deal with saturated pixels. If your star is not saturated, set this
			greater than 1.0.
		consensus: Boolean
			If True, this will subsample the pixel space, separately calculate halo time
			series for eah set of pixels, and merge these at the end. This is to check
			for validation, but is typically not useful, and is by default set False.
		analytic: Boolean
			If True, it will optimize the TV with autograd analytic derivatives, which is
			several orders of magnitude faster than with numerical derivatives. This is
			by default True but you can run it numerically with False if you prefer.
		sigclip: Boolean
			If True, it will iteratively run the TV-min algorithm clipping outliers.
			Use this for data with a lot of outliers, but by default it is set False.
		"""

		# Start logger to use for print
		logger = logging.getLogger(__name__)

		logger.info("starid: %d", self.starid)

		logger.info("Target position in stamp: (%f, %f)", self.target_pos_row_stamp, self.target_pos_column_stamp )

		# Halophot settings:
		splits=(None,None)
		sub = 1
		order = 1
		maxiter = 101
		w_init = None
		random_init = False
		thresh = -1
		minflux = -100
		consensus = False
		analytic = True
		sigclip = False

		# Find timestamps where the timeseries should be split:
		if self.sector == 1:
			split_times = (1339., 1347.366, 1349.315)
		elif self.sector == 2:
			split_times = (1368.)
		else:
			logger.warning("No split-timestamps have been defined for this sector")
			split_times = None # TODO: Is this correct?

		# Initialize
		logger.info('Formatting data for halo')
		flux = self.images_cube.T
		flux[:,self.pixelflags.T==0] = np.nan

		# Get the position of the main target
		col = self.target_pos_column + self.lightcurve['pos_corr'][:, 0]
		row = self.target_pos_row + self.lightcurve['pos_corr'][:, 1]

		# Put together timeseries table in the format that halophot likes:
		ts = Table({
			'time': self.lightcurve['time'],
			'cadence': self.lightcurve['cadenceno'],
			'x': col,
			'y': row,
			'quality': self.lightcurve['quality']
		})

		# Run the halo photometry core function
		try:
			pf, ts, weights, weightmap, pixels_sub = do_lc(
				flux,
				ts,
				splits,
				sub,
				order,
				maxiter=maxiter,
				split_times=split_times,
				w_init=w_init,
				random_init=random_init,
				thresh=thresh,
				minflux=minflux,
				consensus=consensus,
				analytic=analytic,
				sigclip=sigclip
			)

			# Rescale the extracted flux:
			normfactor = mag2flux(self.target['tmag'])/np.nanmedian(ts['corr_flux'])
			self.lightcurve['flux'] = ts['corr_flux'] * normfactor

			# Calculate the flux error by uncertainty propergation:
			for k, imgerr in enumerate(self.images_err):
				self.lightcurve['flux_err'][k] = np.abs(normfactor) * np.sqrt(np.sum( weightmap**2 * imgerr**2 ))

			self.lightcurve['pos_centroid'][:,0] = col # we don't actually calculate centroids
			self.lightcurve['pos_centroid'][:,1] = row

			# Save the weightmap into special property which will make sure
			# that it is saved into the final FITS output files:
			self.halo_weightmap = weightmap

		except:
			logger.exception('Halo optimization failed')
			self.report_details(error='Halo optimization failed')
			return STATUS.ERROR

		# plot
		if self.plot:
			try:
				logger.info('Plotting weight map')
				cmap = plt.get_cmap('seismic')
				norm = np.size(weightmap)
				cmap.set_bad('k', 1.)
				im = np.log10(weightmap*norm)
				fig = plt.figure()
				ax = fig.add_subplot(111)
				plt.imshow(im,cmap=cmap, vmin=-2*np.nanmax(im),vmax=2*np.nanmax(im),
					interpolation='None',origin='lower')
				plt.colorbar()
				ax.set_title('TV-min Weightmap')
				#plot_image(im, scale='log', cmap=cmap, vmin=-2*np.nanmax(im), vmax=2*np.nanmax(im), make_cbar=True, title='TV-min Weightmap')
				save_figure(os.path.join(self.plot_folder, '%sweightmap' % self.starid))
			except:
				logger.exception('Failed to plot')
				self.report_details(error="Failed to plot")
				return STATUS.WARNING

		# If something went seriously wrong:
		#self.report_details(error='What the hell?!')
		#return STATUS.ERROR

		# If something is a bit strange:
		#self.report_details(error='')
		#return STATUS.WARNING

		# Add additional headers specific to this method:
		self.additional_headers['HALO_VER'] = (halophot.__version__, 'Version of halophot')
		self.additional_headers['HALO_ODR'] = (order, 'Halophot nth order TV')
		self.additional_headers['HALO_THR'] = (thresh, 'Halophot saturated pixel threshold')
		self.additional_headers['HALO_MXI'] = (maxiter, 'Halophot maximum optimisation iterations')
		self.additional_headers['HALO_SCL'] = (sigclip, 'Halophot sigma clipping enabled')

		# If some stars could be skipped:
		#self.report_details(skip_targets=skip_targets)

		# Return whether you think it went well:
		return STATUS.OK

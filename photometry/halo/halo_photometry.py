#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Halo Photometry.

.. codeauthor:: Benjamin Pope <benjamin.pope@nyu.edu>
.. codeauthor:: Tim White <white@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
import contextlib
import numpy as np
from astropy.table import Table
import halophot
from halophot.halo_tools import do_lc
from ..plots import plt, plot_image, save_figure
from .. import BasePhotometry, STATUS
from ..quality import TESSQualityFlags
from ..utilities import mag2flux, LoggerWriter

#--------------------------------------------------------------------------------------------------
class HaloPhotometry(BasePhotometry):
	"""
	Use halo photometry to observe very saturated stars.

	.. codeauthor:: Benjamin Pope <benjamin.pope@nyu.edu>
	.. codeauthor:: Tim White <white@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super().__init__(*args, **kwargs)

	#----------------------------------------------------------------------------------------------
	def do_photometry(self):
		"""
		Performs 'halo' TV-min weighted-aperture photometry.

		Parameters
		----------
		aperture_mask : array-like, 'pipeline', or 'all'
			A boolean array describing the aperture such that `False` means
			that the pixel will be masked out.
			If the string 'all' is passed, all pixels will be used.
			The default behavior is to use the Kepler pipeline mask.
		splits : tuple, (None, None) or (2152,2175) etc.
			A tuple including two times at which to split the light curve and run halo
			separately outside these splits.
		sub : int
			Do you want to subsample every nth pixel in your light curve? Not advised,
			but can come in handy for very large TPFs.
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
		objective: string
			Objective function: can be tv, tv_o2, l2v, or l3v.
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

		# Halophot settings:
		splits = (None, None)
		sub = 1
		maxiter = 101
		w_init = None
		random_init = False
		thresh = -1
		minflux = -100.0
		objective = 'tv'
		analytic = True
		sigclip = False
		dist_max = 20.0

		# In the case of FFI, set the size of the postage stamp to be just slightly larger than
		# the maximum distance from the target (allowing for one pixel on each side):
		if self.datasource == 'ffi':
			self.resize_stamp(width=dist_max+2, height=dist_max+2)

		logger.info("Target position in stamp: (%f, %f)",
			self.target_pos_row_stamp,
			self.target_pos_column_stamp)

		# Initialize
		logger.info('Formatting data for halo')
		indx_goodtimes = np.isfinite(self.lightcurve['time'])
		flux = self.images_cube.T[indx_goodtimes, :, :]

		# Cut out pixels closer than 20 pixels, that were actually observed:
		# TODO: We should maybe use self.resize_stamp instead, at least for FFIs.
		# TODO: Should the limit scale with Tmag?
		# TODO: Is there a one pixel offset in dist?
		cols, rows = self.get_pixel_grid()
		dist = np.sqrt((cols - self.target_pos_column)**2 + (rows - self.target_pos_row)**2)
		pixel_mask = (self.aperture & 1 != 0) & (dist <= dist_max)

		# Cut out the pixel mask in flux cube:
		flux[:, ~pixel_mask.T] = np.NaN

		# Find timestamps where the timeseries should be split:
		if self.sector == 1:
			split_times = (1339., 1347.366, 1349.315)
		elif self.sector == 2:
			split_times = (1368.,)
		elif self.sector == 3:
			split_times = (1395.52,)
		elif self.sector == 8:
			split_times = (1529.50,)
		else:
			# If nothing has been explicitely defined for this sector,
			# let's see of there is a large gap somewhere near the middle of
			# timeseries, that is properly due to the data downlink.
			# If a single hole is found, use it, otherwise don't try to be
			# to clever, and just don't split the timeseries.
			timecorr = self.lightcurve['timecorr'][indx_goodtimes]
			t = self.lightcurve['time'][indx_goodtimes] - timecorr
			dt = np.append(np.diff(t), 0)
			t0 = np.nanmin(t)
			Ttot = np.nanmax(t) - t0
			indx = (t0+0.30*Ttot < t) & (t < t0+0.70*Ttot) & (dt > 0.5)
			if np.sum(indx) == 1:
				indx = np.where(indx)[0][0]
				thole = 0.5*(t[indx] + t[indx+1]) + timecorr[indx]
				logger.info("Automatically found split: %f", thole)
				split_times = (thole,)
			else:
				logger.warning("No split-timestamps have been defined for this sector")
				split_times = None # TODO: Is this correct?

		# Get the position of the main target
		col = self.target_pos_column + self.lightcurve['pos_corr'][:, 0]
		row = self.target_pos_row + self.lightcurve['pos_corr'][:, 1]

		# Put together timeseries table in the format that halophot likes:
		ts = Table({
			'time': self.lightcurve['time'][indx_goodtimes],
			'cadence': self.lightcurve['cadenceno'][indx_goodtimes],
			'x': col[indx_goodtimes],
			'y': row[indx_goodtimes],
			'quality': self.lightcurve['quality'][indx_goodtimes]
		})

		# Run the halo photometry core function
		try:
			# Redirect stdout to logger.info
			with contextlib.redirect_stdout(LoggerWriter(logger)):
				pf, ts, weights, weightmap_dict, pixels_sub = do_lc(
					flux,
					ts,
					splits,
					sub,
					maxiter=maxiter,
					split_times=split_times,
					w_init=w_init,
					random_init=random_init,
					thresh=thresh,
					minflux=minflux,
					objective=objective,
					analytic=analytic,
					sigclip=sigclip,
					verbose=logger.isEnabledFor(logging.INFO),
					mission='TESS',
					bitmask=TESSQualityFlags.DEFAULT_BITMASK
				)
		except: # noqa: E722
			logger.exception('Halo optimization failed')
			return STATUS.ERROR

		# Fix for halophot sometimes not returning lists:
		for key, value in weightmap_dict.items():
			if not isinstance(value, list):
				weightmap_dict[key] = [value]

		# Rescale the extracted flux:
		normfactor = mag2flux(self.target['tmag'])
		self.lightcurve['flux'][indx_goodtimes] = ts['corr_flux'] * normfactor

		# Create mapping from each cadence to which weightmap was used:
		wmindx = np.zeros_like(indx_goodtimes, dtype=int)
		for k, (cad1, cad2) in enumerate(zip(weightmap_dict['initial_cadence'], weightmap_dict['final_cadence'])):
			wmindx[(self.lightcurve['cadenceno'] >= cad1) & (self.lightcurve['cadenceno'] <= cad2)] = k

		# Calculate the flux error by uncertainty propergation:
		for k, imgerr in enumerate(self.images_err):
			if not indx_goodtimes[k]: continue
			wm = weightmap_dict['weightmap'][wmindx[k]] # Get the weightmap for this cadence
			self.lightcurve['flux_err'][k] = np.abs(normfactor) * np.sqrt(np.nansum( wm**2 * imgerr**2 ))

		self.lightcurve['pos_centroid'][:,0] = col # we don't actually calculate centroids
		self.lightcurve['pos_centroid'][:,1] = row

		# Save the weightmap into special property which will make sure
		# that it is saved into the final FITS output files:
		self.halo_weightmap = weightmap_dict

		# Plotting:
		if self.plot:
			logger.info('Plotting weight map')
			norm = np.size(weightmap_dict['weightmap'][0])
			for k, wm in enumerate(weightmap_dict['weightmap']):
				im = np.log10(wm*norm)
				fig, ax = plt.subplots()
				plot_image(im, ax=ax, scale='linear', title='TV-min Weightmap', cmap='seismic',
					cbar='right', vmin=-2*np.nanmax(im), vmax=2*np.nanmax(im), clabel=None)
				save_figure(os.path.join(self.plot_folder, '%d_weightmap_%d' % (self.starid, k+1)), fig=fig)
				plt.close(fig)

		# Add additional headers specific to this method:
		self.additional_headers['HALO_VER'] = (halophot.__version__, 'Version of halophot')
		self.additional_headers['HALO_OBJ'] = (objective, 'Halophot objective function')
		self.additional_headers['HALO_THR'] = (thresh, 'Halophot saturated pixel threshold')
		self.additional_headers['HALO_MXI'] = (maxiter, 'Halophot maximum optimisation iterations')
		self.additional_headers['HALO_SCL'] = (sigclip, 'Halophot sigma clipping enabled')
		self.additional_headers['HALO_MFL'] = (minflux, 'Halophot minimum flux')

		# Return mask used for photometry:
		self.final_phot_mask = pixel_mask

		# Check if there are other targets in the mask that could then be skipped from
		# processing, and report this back to the TaskManager. The TaskManager will decide
		# if this means that this target or the other targets should be skipped in the end.
		cols, rows = self.get_pixel_grid()
		skip_targets = [t['starid'] for t in self.catalog if t['starid'] != self.starid
			and np.any(pixel_mask & (rows == np.round(t['row'])+1) & (cols == np.round(t['column'])+1))]
		if skip_targets:
			logger.info("These stars could be skipped: %s", skip_targets)
			self.report_details(skip_targets=skip_targets)

		# Return whether you think it went well:
		return STATUS.OK

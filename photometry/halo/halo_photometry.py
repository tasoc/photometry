#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Halo Photometry.

.. codeauthor:: Tim White <white@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import logging
from .. import BasePhotometry, STATUS
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

		This function needs to set
			* self.lightcurve
		"""

		# Start logger to use for print 
		logger = logging.getLogger(__name__)

		logger.info("starid: %d", self.starid)
		
		logger.info("Target position in stamp: (%f, %f)", self.target_pos_row_stamp, self.target_pos_column_stamp )

		# Get pixel grid:
		cols, rows = self.get_pixel_grid()

		'''
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
		'''

		splits=(None,None)
		sub=1
		order=1
		maxiter=101
		w_init=None
		random_init=False
		thresh=0.8
		minflux=100.
		consensus=False
		analytic=True
		sigclip=False

		logger.info('Formatting TPF data for halo')
		
		flux = self.images_cube.T
		flux[:,self.pixelflags==0] = np.nan
		logger.info("Flux dimension")
		logger.info(flux.shape)
		time = self.lightcurve['time']
		quality = self.lightcurve['quality']
		x, y = self.tpf[1].data['POS_CORR1'], self.tpf[1].data['POS_CORR2']
		
		ts = Table({'time':time,
					'cadence':self.lightcurve['cadenceno'],
					'x':x,
					'y':y,
					'quality':quality})

		# try:
		logger.info('Attempting TV-min photometry')
		pf, ts, weights, weightmap, pixels_sub = do_lc(flux,
					ts,splits,sub,order,maxiter=101,w_init=None,random_init=False,
			thresh=0.8,minflux=100.,consensus=False,analytic=True,sigclip=False)

		self.lightcurve['corr_flux'] = ts['corr_flux']
		self.halo_weightmap = weightmap
		# except: 
		# 	self.report_details(error='Halo optimization failed')
		# 	return STATUS.ERROR

		# plot

		try:
			logger.info('Plotting weight map')
			cmap = mpl.cm.seismic
			norm = np.size(weightmap)
			cmap.set_bad('k',1.)
			im = np.log10(weightmap.T*norm)
			plt.imshow(im,cmap=cmap, vmin=-2*np.nanmax(im),vmax=2*np.nanmax(im),
				interpolation='None',origin='lower')
			plt.colorbar()
			plt.title('TV-min Weightmap')
			plt.savefig('%sweightmap.png' % self.plot_folder)
		except:
			self.report_details(error="Failed to plot")
			return STATUS.WARNING

		# If something went seriouly wrong:
		#self.report_details(error='What the hell?!')
		#return STATUS.ERROR

		# If something is a bit strange:
		#self.report_details(error='')
		#return STATUS.WARNING
		
		# Save the mask to be stored in the outout file:

		# Add additional headers specific to this method:
		#self.additional_headers['HDR_KEY'] = (value, 'comment')

		# If some stars could be skipped:
		#self.report_details(skip_targets=skip_targets)

		# Return whether you think it went well:
		return STATUS.OK


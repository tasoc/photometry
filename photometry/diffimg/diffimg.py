#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Difference Imaging Photometry.

.. codeauthor:: Isabel Colman <isabel.colman@sydney.edu.au>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import logging
from functools import partial
import scipy.ndimage.interpolation as spint
from .. import BasePhotometry, STATUS

#------------------------------------------------------------------------------
class DiffImgPhotometry(BasePhotometry):
   """Simple Aperture Photometry using K2P2 to define masks.

   .. codeauthor:: Isabel Colman <isabel.colman@sydney.edu.au>
   """

   def __init__(self, *args, **kwargs):
      # Call the parent initializing:
      # This will set several default settings
      super(self.__class__, self).__init__(*args, **kwargs)

      # Here you could do other things that needs doing in the beginning
      # of the run on each target.

   def _minimum_aperture(self):
      x = int(self.target_pos_row_stamp + 0.5)
      y = int(self.target_pos_column_stamp - 0.5)
      mask_main = np.zeros_like(self.sumimage, dtype='bool')
      # mask_main[y:y+2, x:x+2] = True # think about re-implementing this if no target pos?
      return mask_main


   def do_photometry(self):
      """Perform photometry on the given target.

      This function needs to set
         * self.lightcurve
      """

      def regrid(flux, factor=2):
         """
         Takes a 2D array and expands it

         The flux array is expanded into a series of factor by factor submatricies. These
         submatricies are arranged into the same ordering as the original array.

         Arguments:
         :: flux (2D ndarray) :: The 2D array to be regridded

         Keyword arguments:
         :: factor (int) :: The scaling factor for the regridding. 
            Default is 2

         Returns:
         :: 2D ndarray :: Containing the regridded values

         Function courtesy of Alan Robertson
         """

         # Increasing function; don't try to multiply lists as it copies the reference 
         inc = lambda factor, value: np.zeros((factor, factor)) + value
         inc = partial(inc, factor)

         # Some useful formats
         block_shape = tuple(list(flux.shape) + [factor, factor])
         final_shape = tuple(np.array(flux.shape) * factor) 
         transpose_ordering = [0,3,1,2]

         # The Map operation 
         re_flux = np.array(list(map(inc, flux.reshape(flux.size))))

         # Reshaping to the correct view
         re_flux = re_flux.reshape(block_shape).transpose(transpose_ordering)
         re_flux = re_flux.reshape(final_shape)
         
         return re_flux

      def centroid(fluxarr, mask):

         # inputs: length of flux array, flux array (3 dimensional), aperture mask for centroiding over only target
         # rows: axis 0, cols: axis 1

         timespan = fluxarr.shape[2]
         x_cent = np.zeros(timespan)
         y_cent = np.zeros(timespan)
         index = np.indices((fluxarr.shape[0],fluxarr.shape[1]))

         for i in range(timespan):
            xfsum = 0
            yfsum = 0
            fsum = 0
            temp = fluxarr[:, :, i]
            xfsum = sum(map((lambda x,y: x*y), index[0,:][np.where(mask==True)], temp[np.where(mask==True)]))
            yfsum = sum(map((lambda x,y: x*y), index[1,:][np.where(mask==True)], temp[np.where(mask==True)]))
            fsum = sum(temp[np.where(mask==True)])
            # for index, val in np.ndenumerate(temp):
            #    if mask[index] == True:
            #       xfsum += index[1] * temp[index]
            #       yfsum += index[0] * temp[index]
            #       fsum += temp[index]
            #    else:
            #       pass
            x_cent[i] = xfsum / fsum
            y_cent[i] = yfsum / fsum

         return x_cent, y_cent

      logger = logging.getLogger(__name__)
      logger.info("Running difference photometry...")

      # self.resize_stamp(up=1) # for testing

      SumImage = self.sumimage # average

      logger.info(self.stamp)
      logger.info("Target position in stamp: (%f, %f)", self.target_pos_row_stamp, self.target_pos_column_stamp )

      cat = np.column_stack((self.catalog['column_stamp'], self.catalog['row_stamp'], self.catalog['tmag']))

      logger.info("Creating new masks...")

      # Masking etc
      centre = (int(round(self.target_pos_row_stamp)), int(round(self.target_pos_column_stamp)))

      # Mask for postage stamp
      mask_main = self._minimum_aperture()
      width = 2 # small default aperture for choosing centroid, before finer aperture is chosen for masking
      mask_main[centre[0]-width:centre[0]+width,centre[1]-width:centre[1]+width] = True

      # XY of pixels in frame
      cols, rows = self.get_pixel_grid()
      members = np.column_stack((cols[mask_main], rows[mask_main]))

      # Targets that are in the mask:
      target_in_mask = [k for k,t in enumerate(self.catalog) if np.round(t['row'])+1 in rows[mask_main] and np.round(t['column'])+1 in cols[mask_main]]

      # Calculate contamination from the other targets in the mask:
      if len(target_in_mask) == 1 and self.catalog[target_in_mask][0]['starid'] == self.starid:
         contamination = 0
      else:
         # Calculate contamination metric as defined in Lund & Handberg (2014):
         mags_in_mask = self.catalog[target_in_mask]['tmag']
         mags_total = -2.5*np.log10(np.nansum(10**(-0.4*mags_in_mask)))
         contamination = 1.0 - 10**(0.4*(mags_total - self.target_tmag))
         contamination = np.abs(contamination) # Avoid stupid signs due to round-off errors

      logger.info("Contamination: %f", contamination)
      self.additional_headers['AP_CONT'] = (contamination, 'AP contamination')

      # If contamination is high, return a warning:
      if contamination > 0.1:
         self.report_details(error='High contamination')
         return STATUS.WARNING

      # Skip targets
      skip_targets = [t['starid'] for t in self.catalog[target_in_mask] if t['starid'] != self.starid]
      if skip_targets:
         logger.info("These stars could be skipped:")
         logger.info(skip_targets)
         self.report_details(skip_targets=skip_targets)

      # Some regridding/masking parameters
      factor = 2 # regridding factor; minimum 2, MUST be even
      ef = factor/2 # expansion factor so that mask always increments in width of 1/2 pixel, ideally chosen based on magnitude??

      # Calculate centroid & shifts
      x_cent, y_cent = centroid(self.images_cube, mask_main)
      x_shifts = (x_cent - x_cent[0])*factor # choosing the zeroth element as the point to shift to
      y_shifts = (y_cent - y_cent[0])*factor

      # Initial regridding for average
      # x = len(cols[0])*factor
      # y = len(rows[0])*factor
      x = mask_main.shape[0]*factor
      y = mask_main.shape[1]*factor

      rg_avg = regrid(SumImage, factor=factor)
      rg_flux = np.array(list(map(partial(regrid, factor=factor), self.images)))
      rg_bg = np.array(list(map(partial(regrid, factor=factor), self.backgrounds)))
      rg_mask = np.zeros_like(rg_avg, dtype='bool')

      # aperture
      maskwidth = 3
      masknum = maskwidth*ef # this number being how many half-pixels wide to make the mask
      rg_mask[int(centre[0]-masknum):int(centre[0]+masknum+ef), int(centre[1]-masknum):int(centre[1]+masknum+ef)] = True
      rg_mask[int(centre[0]-masknum):int(centre[0]-masknum+ef), int(centre[1]-masknum):int(centre[1]-masknum+ef)] = False
      rg_mask[int(centre[0]-masknum):int(centre[0]-masknum+ef), int(centre[1]+masknum):int(centre[1]+masknum+ef)] = False
      rg_mask[int(centre[0]+masknum):int(centre[0]+masknum+ef), int(centre[1]-masknum):int(centre[1]-masknum+ef)] = False
      rg_mask[int(centre[0]+masknum):int(centre[0]+masknum+ef), int(centre[1]+masknum):int(centre[1]+masknum+ef)] = False

      # Realignment
      shifted = np.zeros(rg_flux.shape)

      for i in range(shifted.shape[0]):
         shifted[i,:,:] = spint.shift(rg_flux[i,:,:], (-y_shifts[i], -x_shifts[i]), order=1)

      # Loop through the images and backgrounds together:
      for k, (img, bck) in enumerate(zip(shifted, rg_bg)):

         flux_in_cluster = img[rg_mask]
         
         # Do subtraction
         avg_in_cluster = rg_avg[rg_mask] # this should be done for new postage stamp size as well

         subtracted = flux_in_cluster - avg_in_cluster

         # Calculate flux in mask:
         self.lightcurve['flux'][k] = np.sum(subtracted) + np.sum(avg_in_cluster)
         # self.lightcurve['flux'][k] = np.sum(flux_in_cluster)
         self.lightcurve['flux_background'][k] = np.sum(bck[rg_mask])

      # Save the mask to be stored in the outout file:
      self.final_mask = mask_main

      # Add additional headers specific to this method:
      self.additional_headers['METHOD'] = ('diff', 'Light curve produced by difference imaging')
      self.additional_headers['REGRID'] = (factor, 'Regridding factor')
      self.additional_headers['MASKSIZE'] = (maskwidth, 'Max mask width in half-pixels')

      # Return whether you think it went well:
      return STATUS.OK
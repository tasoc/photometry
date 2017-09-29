#!/bin/env python
# -*- coding: utf-8 -*-

"""K2 Pixel Photometry (K2P2)

Create pixel masks and extract light curves from Kepler and K2 pixel data
using clustering algorithms.

@version: $Revision: 723 $
@author:  Rasmus Handberg & Mikkel Lund ($Author: rasmush $)
@date:    $Date: 2016-09-28 10:47:06 +0200 (Wed, 28 Sep 2016) $"""

#==============================================================================
# TODO
#==============================================================================
# * If number of clusters > max_cluster then only save max_cluster largest
# * What to do if more clusters associate with same known star?
# * more use of quality flags when they arrive
# * uniqueness of pixel-cluster-membership when extending overflow columns
# * wcs routine
# * estimate magnitude of other stars

#==============================================================================
# Packages
#==============================================================================

from __future__ import division, with_statement
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
y_formatter = ScalarFormatter(useOffset=False)
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from statsmodels.nonparametric.bandwidths import select_bandwidth
from scipy import ndimage
import os
from scipy import optimize as OP
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from bottleneck import nanmedian
import logging
from re import match

#==============================================================================
# Constants and settings
#==============================================================================

# The version of K2P2:
__version__ = '1.1 - build %d' % int(match(r'\$Revision:\s*(\d+)\s*\$', '$Revision: 723 $').group(1))

# Constants:
mad_to_sigma = 1.482602218505602 # Constant is 1/norm.ppf(3/4)

# Custom exceptions we may raise:
class K2P2NoFlux(Exception):
	pass

#==============================================================================
# Mask outline
#==============================================================================
def k2p2maks(frame, no_combined_images, threshold=0.5):

	thres_val = no_combined_images * threshold
	mapimg = (frame > thres_val)
	ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])
	hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

	l = []
	for p in zip(*hor_seg):
		l.append((p[1], p[0]+1))
		l.append((p[1]+1, p[0]+1))
		l.append((np.nan,np.nan))

	# and the same for vertical segments
	for p in zip(*ver_seg):
		l.append((p[1]+1, p[0]))
		l.append((p[1]+1, p[0]+1))
		l.append((np.nan, np.nan))


	segments = np.array(l)

	x0 = -0.5
	x1 = frame.shape[1]+x0
	y0 = -0.5
	y1 = frame.shape[0]+y0

	# now we need to know something about the image which is shown
	#   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
	#   drawn with origin='lower'
	# with this information we can rescale our points
	segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
	segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

	return segments

#==============================================================================
# Configure pixel plots
#==============================================================================
def config_pixel_plot(ax, size=None, aspect='auto'):
	"""Utility function for configuring axes when plotting pixel images."""

	if not size is None:
		ax.set_xlim([-0.5, size[1]-0.5])
		ax.set_ylim([-0.5, size[0]-0.5])

	ax.xaxis.set_major_locator(MultipleLocator(5))
	ax.xaxis.set_minor_locator(MultipleLocator(1))
	ax.yaxis.set_major_locator(MultipleLocator(5))
	ax.yaxis.set_minor_locator(MultipleLocator(1))
	ax.tick_params(direction='out', which='both', pad=5)
	ax.xaxis.tick_bottom()
	ax.set_aspect(aspect)

#==============================================================================
# DBSCAN subroutine
#==============================================================================
def run_DBSCAN(X2, Y2, cluster_radius, min_for_cluster):

	XX = np.array([[x,y] for x,y in zip(X2,Y2)])

	db = DBSCAN(eps=cluster_radius, min_samples=min_for_cluster)
	db.fit(XX)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	# Number of clusters in labels, ignoring noise if present.

	return XX, labels, core_samples_mask

#==============================================================================
# Segment clusters using watershed
#==============================================================================
def k2p2WS(X, Y, X2, Y2, flux0, XX, labels, core_samples_mask, saturated_masks=None, ws_thres=0.1, ws_footprint=3, ws_blur=0.5, ws_alg='flux', output_folder=None, catalog=None):

	# Get logger for printing messages:
	logger = logging.getLogger(__name__)


	unique_labels_ini = set(labels)


	XX2 = np.array([[x,y] for x,y in zip(X.flatten(),Y.flatten())])

	Labels = np.ones_like(flux0)*-2
	Labels[XX[:,1], XX[:,0]] = labels

	Core_samples_mask = np.zeros_like(Labels, dtype=bool)
	Core_samples_mask[XX[:,1], XX[:,0]] = core_samples_mask

	# Set all non-core points to noise
	Labels[~Core_samples_mask] = -1

	max_label = np.max(labels)

	for i in xrange(len(unique_labels_ini)):


		lab = list(unique_labels_ini)[i]

		if lab == -1 or lab == -2:
			continue

		# select class members - non-core members have been set to noise
		class_member_mask = (Labels == lab).flatten()
		xy = XX2[class_member_mask,:]


		Z = np.zeros_like(flux0, dtype='float64')
		Z[xy[:,1], xy[:,0]] = flux0[xy[:,1], xy[:,0]] #y=row, x=column

		if ws_alg == 'dist':
			distance0 = ndimage.distance_transform_edt(Z)
		elif ws_alg == 'flux':
			distance0 = Z
		else:
			logger.error("Unknown watershed algorithm: '%s'", ws_alg)

		logger.debug("Using '%s' watershed algorithm", ws_alg)

		if not catalog is None:
			distance = distance0

			local_maxi = np.zeros_like(flux0, dtype='bool')
			#for c in catalog:
			#	local_maxi[int(np.floor(c[1])), int(np.floor(c[0]))] = True

			#print("Finding blobs...")
			#blobs = blob_dog(distance, min_sigma=1, max_sigma=20)
			#for c in blobs:
			#	local_maxi[int(np.floor(c[0])), int(np.floor(c[1]))] = True

			# Smooth the basin image with Gaussian filter:
			#distance = ndimage.gaussian_filter(distance0, ws_blur*0.5)

			# Find maxima in the basin image to use for markers:
			local_maxi_loc = peak_local_max(distance, indices=True, exclude_border=False, threshold_rel=0, footprint=np.ones((ws_footprint, ws_footprint)))

			for c in catalog:
				d = np.sqrt( ((local_maxi_loc[:,1]+0.5) - c[0])**2 + ((local_maxi_loc[:,0]+0.5) - c[1])**2 )
				indx = np.argmin(d)
				if d[indx] < 2.0*np.sqrt(2):
					local_maxi[local_maxi_loc[indx,0], local_maxi_loc[indx,1]] = True

		else:
			# Smooth the basin image with Gaussian filter:
			distance = ndimage.gaussian_filter(distance0, ws_blur)

			# Find maxima in the basin image to use for markers:
			local_maxi = peak_local_max(distance, indices=False, exclude_border=False, threshold_rel=ws_thres, footprint=np.ones((ws_footprint, ws_footprint)))

		# If masks of saturated pixels are provided, clean out in the
		# found local maxima to make sure only one is found within
		# each patch of saturated pixels:
		if not saturated_masks is None and lab in saturated_masks:
			saturated_pixels = saturated_masks[lab]

			# Split the saturated pixels up into patches that are connected:
			sat_labels, numfeatures = ndimage.label(saturated_pixels)

			# Loop through the patches of saturated pixels:
			for k in xrange(1, numfeatures+1):
				# This mask of saturated pixels:
				sp = saturated_pixels & (sat_labels == k)

				# Check if there is more than one local maximum found
				# within this patch of saturated pixels:
				if np.sum(local_maxi & sp) > 1:
					# Find the local maximum with the highest value that is also saturated:
					imax = np.unravel_index(np.nanargmax(distance * local_maxi * sp), distance.shape)
					# Only keep the maximum with the highest value and remove all
					# the others if they are saturated:
					local_maxi[sp] = False
					local_maxi[imax] = True

		# Assign markers/labels to the found maxima:
		markers = ndimage.label(local_maxi)[0]

		# Run the watershed segmentation algorithm on the negative
		# of the basin image:
		labels_ws = watershed(-distance0, markers, mask=Z)

		# The number of masks after the segmentation:
		no_labels = len(set(labels_ws.flatten()))

		# Set all original cluster points to noise, in this way things that in the
		# end is not associated with a "new" cluster will not be used any more
		Labels[xy[:,1], xy[:,0]] = -1

		# Use the original label for a part of the new cluster -  if only
		# one cluster is identified by the watershed algorithm this will then
		# keep the original labeling
		idx = (labels_ws==1) & (Z!=0)
		Labels[idx] = lab

		# If the cluster is segmented we will assign these new labels, starting from
		# the highest original label + 1
		for u in xrange(no_labels-2):
			max_label += 1

			idx = (labels_ws==u+2) & (Z!=0)
			Labels[idx] = max_label

		labels_new = Labels[Y2, X2]
		unique_labels = set(labels_new)
		NoCluster = len(unique_labels) - (1 if -1 in labels_new else 0)

		# Create plot of the watershed segmentation:
		if not output_folder is None:

			fig, axes = plt.subplots(ncols=3, figsize=(14, 6))
			fig.subplots_adjust(hspace=0.12, wspace=0.12)
			ax0, ax1, ax2 = axes

			ax0.imshow(np.log10(Z), cmap=plt.cm.Blues, interpolation='none', origin='lower')
			ax0.set_title('Overlapping objects')

			ax1.imshow(np.log10(distance), cmap=plt.cm.Blues, interpolation='nearest', origin='lower')
			if not catalog is None:
				ax1.scatter(catalog[:,0], catalog[:,1], color='y', s=5, alpha=0.3)
			ax1.scatter(X[local_maxi], Y[local_maxi], color='r', s=5, alpha=0.5)
			ax1.set_title('Basin')

			ax2.imshow(labels_ws, cmap=plt.cm.spectral, interpolation='nearest', origin='lower')
			ax2.set_title('Separated objects')

			for ax in axes:
				config_pixel_plot(ax, size=flux0.shape, aspect='equal')
				ax.set_xticklabels([])
				ax.set_yticklabels([])

			figname = 'seperated_cluster_%d.png' % i
			fig.savefig(os.path.join(output_folder, figname), format='png', bbox_inches='tight', dpi=200)
			plt.close(fig)

	return labels_new, unique_labels, NoCluster

#==============================================================================
#
#==============================================================================
def k2p2_saturated(SumImage, MASKS, idx):

	# Get logger for printing messages:
	logger = logging.getLogger(__name__)

	no_masks = MASKS.shape[0]

	column_mask = np.zeros_like(SumImage, dtype='bool')
	saturated_mask = np.zeros_like(MASKS, dtype='bool')
	pixels_added = 0

	# Loop through the different masks:
	for u in xrange(no_masks):
		# Create binary version of mask and extract
		# the rows and columns which it spans and
		# the highest value in it:
		mask = np.asarray(MASKS[u, :, :], dtype='bool')
		mask_rows, mask_columns = np.where(mask)
		mask_max = np.nanmax(SumImage[mask])

		# Loop through the columns of the mask:
		for c in set(mask_columns):

			column_mask[:, c] = True

			# Extract the pixels that are in this column and in the mask:
			pixels = SumImage[mask & column_mask]

			# Calculate ratio as defined in Lund & Handberg (2014):
			ratio = np.abs(nanmedian(np.diff(pixels)))/np.nanmax(pixels)
			if ratio < 0.01 and nanmedian(pixels) >= mask_max/2:
				logger.debug("Column %d - RATIO = %f - Saturated", c, ratio)

				# Has significant flux and is in saturated column:
				add_to_mask = (idx & column_mask)

				# Make sure the pixels we add are directly connected to the highest flux pixel:
				new_mask_labels, numfeatures = ndimage.label(add_to_mask)
				imax = np.unravel_index(np.nanargmax(SumImage * mask * column_mask), SumImage.shape)
				add_to_mask &= (new_mask_labels == new_mask_labels[imax])

				# Modify the mask:
				pixels_added += np.sum(add_to_mask) - np.sum(mask[column_mask])
				logger.debug("  %d pixels should be added to column %d", np.sum(add_to_mask) - np.sum(mask[column_mask]), c)
				saturated_mask[u][add_to_mask] = True
			else:
				logger.debug("Column %d - RATIO = %f", c, ratio)

			column_mask[:, c] = False

	return saturated_mask, pixels_added

#==============================================================================
#
#==============================================================================
def k2p2FixFromSum(SumImage, pixfile, thresh=1, output_folder=None, plot_folder=None, plot=True,show_plot=True,
				   min_no_pixels_in_mask=8, min_for_cluster=4, cluster_radius=np.sqrt(2),
				   segmentation=True, ws_alg='flux', ws_blur=0.5, ws_thres=0.05, ws_footprint=3,
				   extend_overflow=True, catalog=None):

	# Get logger for printing messages:
	logger = logging.getLogger(__name__)
	logger.info("Creating masks from sum-image...")

	if pixfile is None:
		NY, NX = np.shape(SumImage)
		ori_mask = ~np.isnan(SumImage)
	else:
		ori_mask = pixfile[2].data > 0
		NX = pixfile[2].header['NAXIS1']
		NY = pixfile[2].header['NAXIS2']
	X, Y = np.meshgrid(np.arange(NX), np.arange(NY))

	# Cut out pixels from sum image which were collected and contains flux
	# and flatten the 2D image to 1D array:
	Flux = SumImage[ori_mask].flatten()
	Flux = Flux[Flux > 0]

	# Check if there was actually any flux measured:
	if len(Flux) == 0:
		raise K2P2NoFlux("No measured flux in sum-image")

	# Cut away the top 10% of the fluxes:
	flux_cut = stats.trim1(np.sort(Flux), 0.15)

	# Estimate the bandwidth we are going to use for the background:
	background_bandwidth = select_bandwidth(flux_cut, bw='scott', kernel='gau')
	logger.debug("  Sum-image KDE bandwidth: %f", background_bandwidth)

	# Make the Kernel Density Estimation of the fluxes:
	kernel = KDE(flux_cut)
	kernel.fit(kernel='gau', bw=background_bandwidth, fft=True, gridsize=100)

	# MODE
	def kernel_opt(x): return -1*kernel.evaluate(x)
	max_guess = kernel.support[np.argmax(kernel.density)]
	MODE = OP.fmin_powell(kernel_opt, max_guess, disp=0)

	# MAD (around mode)
	MAD1 = mad_to_sigma * nanmedian( np.abs( Flux[(Flux < MODE)] - MODE ) )

	# Define the cutoff above which pixels are regarded significant:
	CUT = MODE + thresh * MAD1

	logger.debug("  Threshold used: %f", thresh)
	logger.debug("  Flux cut is: %f", CUT)
	if logger.isEnabledFor(logging.DEBUG) and not plot_folder is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.fill_between(kernel.support, kernel.density, alpha=0.3)
		ax.axvline(MODE, color='k')
		ax.axvline(CUT, color='r')
		ax.set_xlabel('Flux')
		ax.set_ylabel('Distribution')
		fig.savefig(os.path.join(plot_folder, 'flux_distribution.png'), format='png', bbox_inches='tight')
		plt.close(fig)

	#==========================================================================
	# Find and seperate clusters of pixels
	#==========================================================================

	# Cut out pixels of sum image with flux above the cut-off:
	idx = (SumImage > CUT)
	X2 = X[idx]
	Y2 = Y[idx]

	logger.debug("  Min for cluster is: %f", min_for_cluster)
	logger.debug("  Cluster radius is: %f", cluster_radius)

	# Run clustering algorithm
	XX, labels_ini, core_samples_mask = run_DBSCAN(X2, Y2, cluster_radius, min_for_cluster)

	# Run watershed segmentation algorithm:
	# Demand that there was any non-noise clusters found.
	if segmentation and any(labels_ini != -1):
		# Create a set of dummy-masks that are made up of the clusters
		# that were found by DBSCAN, meaning that there could be masks
		# with several stars in them:
		DUMMY_MASKS = np.zeros((0, NY, NX), dtype='bool')
		DUMMY_MASKS_LABELS = []
		m = np.zeros_like(SumImage, dtype='bool')
		for lab in set(labels_ini):
			if lab == -1: continue
			# Create "image" of this mask:
			m[:,:] = False
			for x,y in XX[labels_ini == lab]:
				m[y, x] = True
			# Append them to lists:
			DUMMY_MASKS = np.append(DUMMY_MASKS, [m], axis=0)
			DUMMY_MASKS_LABELS.append(lab)

		# Run the dummy masks through the detection of saturated columns:
		logger.debug("Detecting saturated columns in non-segmentated masks...")
		smask, _ = k2p2_saturated(SumImage, DUMMY_MASKS, idx)

		# Create dictionary that will map a label to the mask of saturated pixels:
		if np.any(smask):
			saturated_masks = {}
			for u,sm in enumerate(smask):
				saturated_masks[DUMMY_MASKS_LABELS[u]] = sm
		else:
			saturated_masks = None

		# Run the mask segmentaion algorithm on the found clusters:
		labels, unique_labels, NoCluster = k2p2WS(X, Y, X2, Y2, SumImage, XX, labels_ini, core_samples_mask, saturated_masks=saturated_masks, ws_thres=ws_thres,
												  ws_footprint=ws_footprint, ws_blur=ws_blur, ws_alg=ws_alg, output_folder=plot_folder, catalog=catalog)
	else:
		labels = labels_ini
		unique_labels = set(labels)
		NoCluster = len(unique_labels) - (1 if -1 in labels else 0)

	# Make sure it is a tuple and not a set - much easier to work with:
	unique_labels = tuple(unique_labels)

	# Create list of clusters and their number of pixels:
	No_pix_sort = np.zeros([len(unique_labels), 2])
	for u,lab in enumerate(unique_labels):
		No_pix_sort[u, 0] = np.sum(labels == lab)
		No_pix_sort[u, 1] = lab

	# Only select the clusters that have enough pixels and are not noise:
	cluster_select = (No_pix_sort[:, 0] >= min_no_pixels_in_mask) & (No_pix_sort[:, 1] != -1)
	no_masks = sum(cluster_select)
	No_pix_sort = No_pix_sort[cluster_select, :]

	# No masks were found, so return None:
	if no_masks == 0:
		MASKS = None

	else:
		# Sort the clusters by the number of pixels:
		cluster_sort = np.argsort(No_pix_sort[:, 0])
		No_pix_sort = No_pix_sort[cluster_sort[::-1], :]

		# Create 3D array that will hold masks for each target:
		MASKS = np.zeros((no_masks, NY, NX))
		for u in xrange(no_masks):
			lab = No_pix_sort[u, 1]
			class_member_mask = (labels == lab)
			xy = XX[class_member_mask ,:]
			MASKS[u, xy[:,1], xy[:,0]] = 1

		#==========================================================================
		# Fill holes in masks
		#==========================================================================
		pattern = np.array([[[0, 0.25, 0],[0.25, 0, 0.25],[0, 0.25, 0]]]) # 3D array - shape=(1, 3, 3)
		mask_holes_indx = ndimage.convolve(MASKS, pattern, mode='constant', cval=0.0)
		mask_holes_indx = (mask_holes_indx > 0.95) & (MASKS == 0) # Should be exactly 1.0, but let's assume some round-off errors
		if np.any(mask_holes_indx):
			logger.info("Filling %d holes in the masks", np.sum(mask_holes_indx))
			MASKS[mask_holes_indx] = 1

			if not plot_folder is None:
				# Create image showing all masks at different levels:
				img = np.zeros((NY,NX))
				for r in np.transpose(np.where(MASKS > 0)):
					img[r[1], r[2]] = r[0]+1

				# Plot everything together:
				fig = plt.figure()
				ax = fig.add_subplot(111)
				ax.matshow(img, origin='lower', cmap=cm.Spectral, interpolation='none')
				ax.set_xlabel("X pixels")
				ax.set_ylabel("Y pixels")
				ax.set_title("Holes in mask filled")
				config_pixel_plot(ax, size=img.shape, aspect='equal')

				# Create outline of filled holes:
				for hole in np.transpose(np.where(mask_holes_indx)):
					cen = (hole[2]-0.5, hole[1]-0.5)
					ax.add_patch(mpl.patches.Rectangle(cen, 1, 1, color='k', lw=2, fill=False, hatch='//'))

				fig.savefig(os.path.join(plot_folder, 'mask_filled_holes.png'), format='png', bbox_inches='tight')
				plt.close(fig)

		#==========================================================================
		# Entend overflow lanes
		#==========================================================================
		if extend_overflow:
			logger.debug("Detecting saturated columns in masks...")

			# Find pixels that are saturated in each mask and find out if they should
			# be added to the mask:
			saturated_mask, pixels_added = k2p2_saturated(SumImage, MASKS, idx)
			logger.info("Overflow will add %d pixels in total to the masks.", pixels_added)

			# If we are going to plot later on, make a note
			# of how the outline of the masks looked before
			# changinng anything:
			if logger.isEnabledFor(logging.DEBUG):
				outline_before = []
				for u in xrange(no_masks):
					outline_before.append( k2p2maks(MASKS[u,:,:], 1, 0.5) )

			# Add the saturated pixels to the masks:
			MASKS[saturated_mask] = 1

			# If we are running as DEBUG, output some plots as well:
			if logger.isEnabledFor(logging.DEBUG):
				logger.debug("Plotting overflow figures...")
				Ypixel = np.arange(NY)
				for u in xrange(no_masks):
					mask = np.asarray(MASKS[u, :, :], dtype='bool')
					mask_rows, mask_columns = np.where(mask)
					mask_max = np.nanmax(SumImage[mask])

					# The outline of the mask after saturated columns have been
					# corrected for:
					outline = k2p2maks(mask, 1, 0.5)

					with PdfPages(os.path.join(plot_folder, 'overflow_mask' + str(u) + '.pdf')) as pdf:
						for c in sorted(set(mask_columns)):

							column_rows = mask_rows[mask_columns == c]

							title = "Mask %d - Column %d" % (u, c)
							if np.any(saturated_mask[u,:,c]):
								title += " - Saturated"

							fig = plt.figure(figsize=(14,6))
							ax1 = fig.add_subplot(121)
							ax1.axvspan(np.min(column_rows)-0.5, np.max(column_rows)+0.5, color='0.7')
							ax1.plot(Ypixel, SumImage[:, c], 'ro-', drawstyle='steps-mid')
							ax1.set_title(title)
							ax1.set_xlabel('Y pixels')
							ax1.set_ylabel('Sum-image counts')
							ax1.set_ylim(0, mask_max)
							ax1.set_xlim(-0.5, NY-0.5)

							ax2 = fig.add_subplot(122)
							ax2.imshow(np.log10(SumImage), origin='lower', interpolation='none', cmap=cm.Blues)
							ax2.plot(outline_before[u][:,0], outline_before[u][:,1], 'r:')
							ax2.plot(outline[:,0], outline[:,1], 'r-')
							ax2.axvline(c, color='r', ls='--')
							config_pixel_plot(ax2, size=SumImage.shape)

							pdf.savefig(fig)
							plt.close(fig)

	#==============================================================================
	# Create plots
	#==============================================================================
	if not plot_folder is None:
		# Colors to use for each cluster label:
		colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(unique_labels)))

		# Colormap to use for clusters:
		# https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib/9708079#9708079
		#cmap = mpl.colors.ListedColormap(np.append([[1, 1, 1, 1]], colors, axis=0))
		#cmap_norm = mpl.colors.BoundaryNorm(np.arange(-1, len(unique_labels)-1)+0.5, cmap.N)

		# Set up figure to hold subplots:
		if NY/NX > 5:
			aspect = 0.5
		else:
			aspect = 0.2

		fig0 = plt.figure(figsize=(2*plt.figaspect(aspect)))
		fig0.subplots_adjust(wspace=0.12)

		# ---------------
		# PLOT 1
		color_max = np.nanmax(Flux)
		color_min = np.nanmin(Flux)
		Flux_mat = np.log10(SumImage)/np.log10((color_max-color_min))*1.2
		Flux_mat[Flux_mat < 0] = 0

		ax0 = fig0.add_subplot(151)
		ax0.matshow(Flux_mat, origin='lower', cmap=cm.Blues, interpolation='none')
		ax0.set_title("Sum-image")
		config_pixel_plot(ax0, size=SumImage.shape)

		# ---------------
		# PLOT 2
		Flux_mat2 = np.zeros_like(SumImage)
		Flux_mat2[SumImage < CUT] = 1
		Flux_mat2[SumImage > CUT] = 2
		Flux_mat2[ori_mask == 0] = 0

		ax2 = fig0.add_subplot(152)
		ax2.matshow(Flux_mat2, origin='lower', cmap=cm.Spectral, interpolation='none')
		ax2.set_title("Significant flux")
		config_pixel_plot(ax2, size=SumImage.shape)

		# ---------------
		# PLOT 3
		ax2 = fig0.add_subplot(153)
		ax2.set_title("Clustering + Watershed")
		config_pixel_plot(ax2, size=SumImage.shape)

		Flux_mat4 = np.zeros_like(SumImage)
		for u,lab in enumerate(unique_labels):
			class_member_mask = (labels == lab)
			xy = XX[class_member_mask,:]
			if lab == -1:
				# Black used for noise.
				ax2.plot(xy[:, 0], xy[:, 1], '+', markerfacecolor='k',
					 markeredgecolor='k', markersize=5)

			else:
				Flux_mat4[xy[:,1], xy[:,0]] = u+1

				ax2.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=colors[u],
						 markeredgecolor='k', markersize=5)

		# ---------------
		# PLOT 4
		ax4 = fig0.add_subplot(154)
		ax4.matshow(Flux_mat4, origin='lower', cmap=cm.Spectral, interpolation='none')
		ax4.set_title("Extracted clusters")
		config_pixel_plot(ax4, size=SumImage.shape)

		# ---------------
		# PLOT 5un
		ax1 = fig0.add_subplot(155)
		ax1.matshow(Flux_mat, origin='lower', cmap=cm.Blues, interpolation='none')
		ax1.set_title("Final masks")
		config_pixel_plot(ax1, size=SumImage.shape)

		# Plot outlines of selected masks:
		for u in xrange(no_masks):
			# Get the color associated with this label:
			col = colors[ int(np.where(unique_labels == No_pix_sort[u, 1])[0]) ]
			# Make mask outline:
			outline = k2p2maks(MASKS[u, :, :], 1, threshold=0.5)
			# Plot outlines:
			ax1.plot(outline[:, 0], outline[:, 1], color=col, zorder=10, lw=2.5)
			ax4.plot(outline[:, 0], outline[:, 1], color='k', zorder=10, lw=1.5)

		# Save the figure and close it:
		fig0.savefig(os.path.join(plot_folder, 'masks_'+ws_alg+'.png'), format='png', bbox_inches='tight')
		if show_plot:
			plt.show()
		else:
			plt.close('all')

	return MASKS, background_bandwidth

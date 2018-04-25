#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOMF-inspired mask for residual aperture photometry.

@author: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""
import numpy as np


def four_pixel_mask(row, col):
	"""
	Find the indices of a four pixel square mask given a positive subpixel row
	and column star position.

	It is assumed that the coordinates are pixel edge based rather than pixel
	center based.

	Parameters:
		row (float): Subpixel row position of star. Must be positive.
		col (float): subpixel column position of star. Must be positive.

	Returns:
		mask (numpy array, dtype=int): 2D indices of the four pixels in the mask. Convert to 1D using NumPy's ravel_multi_index.
	"""

	if row % 1 < 0.5:
		row_offset = np.array([-1, 0], dtype=int)
	elif row % 1 >= 0.5:
		row_offset = np.array([0, 1], dtype=int)

	rows = np.array([int(row), int(row)], dtype=int)
	rows += row_offset


	if col % 1 < 0.5:
		col_offset = np.array([-1, 0], dtype=int)
	elif col % 1 >= 0.5:
		col_offset = np.array([0, 1], dtype=int)

	cols = np.array([int(col), int(col)], dtype=int)
	cols += col_offset

	mask = np.array([
			[rows[0], cols[0]],
			[rows[0], cols[1]],
			[rows[1], cols[0]],
			[rows[1], cols[1]],
			], dtype=int)

	return mask


if __name__=="__main__":
	# Plot examples of how the residual mask is defined:

	import matplotlib.pyplot as plt

	plt.close('all')

	offset = 0.2
	positions = [(1.+offset, 1.+offset),
				(1.+offset, 2.-offset),
				(2.-offset, 1.+offset),
				(2.-offset, 2.-offset)]

	fig, ax_arr = plt.subplots(1,4)

	for position, ax in zip(positions, ax_arr):
		row, col = position

		mask = four_pixel_mask(row, col)
		print(row, col)
		print(mask)

		image = np.zeros([3, 3], dtype=int)

		for idx in mask:
			image[idx[0], idx[1]] = 1

		im = ax.imshow(image, cmap='Greys_r', origin='lower')
		ax.scatter(col-0.5, row-0.5, color='r')

		plt.tight_layout(pad=1.2)

		# Major ticks
		ax.set_xticks(np.arange(0, 3, 1));
		ax.set_yticks(np.arange(0, 3, 1));

		# Labels for major ticks
		ax.set_xticklabels(np.arange(0, 3, 1));
		ax.set_yticklabels(np.arange(0, 3, 1));

		# Minor ticks
		ax.set_xticks(np.arange(-.5, 3, 1), minor=True);
		ax.set_yticks(np.arange(-.5, 3, 1), minor=True);

		# Gridlines based on minor ticks
		ax.grid(which='minor', color='grey', linestyle='-', linewidth=2)

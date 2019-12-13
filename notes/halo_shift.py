#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sqlite3
import os.path

#------------------------------------------------------------------------------
def mag2flux(mag):
	"""
	Convert from magnitude to flux using scaling relation from
	aperture photometry. This is an estimate.

	Parameters:
		mag (float): Magnitude in TESS band.

	Returns:
		float: Corresponding flux value
	"""
	return 10**(-0.4*(mag - 20.54))

if __name__ == '__main__':
	pass

	folder = r'C:\Users\au195407\Documents\tess_data_local\S01_DR01-2114872'

	conn = sqlite3.connect(os.path.join(folder, 'todo.sqlite'))
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	cursor.execute("SELECT todolist.starid,tmag,onedge,edgeflux FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority;")
	results = cursor.fetchall()

	starid = np.array([row['starid'] for row in results], dtype='int64')
	tmag = np.array([row['tmag'] for row in results])
	OnEdge = np.array([np.NaN if row['onedge'] is None else row['onedge'] for row in results])
	EdgeFlux = np.array([np.NaN if row['edgeflux'] is None else row['edgeflux'] for row in results])

	cursor.close()
	conn.close()

	print(tmag)
	print(OnEdge)
	print(EdgeFlux)

	tmag_limit = 3.0
	flux_limit = 1e-3

	indx = (OnEdge > 0)

	indx_halo = (tmag <= tmag_limit) & (OnEdge > 0) & (EdgeFlux/mag2flux(tmag) > flux_limit)
	indx_spec = (starid == 382420379)

	print(starid[indx_halo])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.scatter(tmag[indx], OnEdge[indx], alpha=0.5)
	plt.scatter(tmag[indx_halo], OnEdge[indx_halo], marker='x', c='r')
	plt.xlim(xmax=tmag_limit)
	plt.ylim(ymin=0)
	ax.set_xlabel('Tmag')


	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(tmag[indx], EdgeFlux[indx], alpha=0.5)
	ax.set_xlim(xmax=5.0)
	#ax.set_ylim(ymin=0.0)
	ax.set_yscale('log')
	ax.set_xlabel('Tmag')

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.scatter(tmag[indx], EdgeFlux[indx]/mag2flux(tmag[indx]), alpha=0.5)
	plt.scatter(tmag[indx_halo], EdgeFlux[indx_halo]/mag2flux(tmag[indx_halo]), alpha=0.3, marker='x', c='r')
	plt.scatter(tmag[indx_spec], EdgeFlux[indx_spec]/mag2flux(tmag[indx_spec]), alpha=0.3, marker='o', c='g', lw=2)

	plt.plot([2.0, 6.0], [1e-3, 2e-2], 'r--')

	plt.axhline(flux_limit, c='r', ls='--')
	plt.axvline(tmag_limit, c='r', ls='--')
	#plt.xlim(xmax=tmag_limit)
	ax.set_ylim(ymin=1e-5, ymax=1)
	ax.set_yscale('log')
	ax.set_ylabel('Edge Flux / Expected Total Flux')
	ax.set_xlabel('Tmag')


	plt.show()
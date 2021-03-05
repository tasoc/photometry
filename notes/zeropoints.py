#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == '__main__':
	# Zeropoints:
	zp = np.array([
		[1, 20.4309, 20.4929],
		[2, 20.4015, 20.4262],
		[3, 20.4433, 20.4587],
		[4, 20.4868, 20.5063],
		[5, 20.5009, 20.4316],
		#[6, 20.6514, 20.5903],
	])

	sector = zp[:,0]

	m = np.median(zp[:, 1:], axis=0)
	s = np.std(zp[:, 1:], axis=0)
	print( m )
	print( s )

	m2 = np.median(zp[:, 1:])
	s2 = np.std(zp[:, 1:])
	print(m2)
	print(s2)

	low = np.min(zp[:, 1:])
	high = np.max(zp[:, 1:])

	print('{0:.3f}--{1:.3f}'.format(low, high))
	range1 = high - low
	range2 = 20.4735 - 20.318
	print(range1)
	print(range2)
	print(range1 / range2)

	fig, ax = plt.subplots()
	ax.plot(sector, zp[:,1], '.-', label='FFI')
	#plt.fill_between([sector[0], sector[-1]], [m[0]+s[0], m[0]+s[0]], [m[0]-s[0], m[0]-s[0]], alpha=0.2)

	ax.plot(sector, zp[:,2], '.-', label='TPF')
	#plt.axhline(m[1], ls='--')
	#plt.axhline(m[1], ls='--')

	ax.axhline(m2, ls='--')
	ax.axhline(m2-s2, ls=':')
	ax.axhline(m2+s2, ls=':')

	ax.set_xlabel('Sector')
	ax.set_ylabel('$zp$')
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.legend()
	plt.show()

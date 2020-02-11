#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize the time coverage of the loaded SPICE kernels.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import glob
import subprocess
import re
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.spice import TESS_SPICE

if __name__ == '__main__':
	plt.switch_backend('Qt5Agg')

	kernel_folder = os.path.abspath('../photometry/data/spice')

	#files = glob.glob(os.path.join(kernel_folder, '*.bsp'))
	files = [os.path.join(kernel_folder, k) for k in TESS_SPICE.kernel_files if k.endswith('.bsp')]

	# List of sectors from tess.mit.edu:
	# WARNING: These are in UTC? The kernels are in ET! We are being a little inconsistent!
	sectors = [
		[0,  datetime.strptime('04/18/18', '%m/%d/%y'), datetime.strptime('07/25/18', '%m/%d/%y')],
		[1,  datetime.strptime('07/25/18', '%m/%d/%y'), datetime.strptime('08/22/18', '%m/%d/%y')],
		[2,  datetime.strptime('08/22/18', '%m/%d/%y'), datetime.strptime('09/20/18', '%m/%d/%y')],
		[3,  datetime.strptime('09/20/18', '%m/%d/%y'), datetime.strptime('10/18/18', '%m/%d/%y')],
		[4,  datetime.strptime('10/18/18', '%m/%d/%y'), datetime.strptime('11/15/18', '%m/%d/%y')],
		[5,  datetime.strptime('11/15/18', '%m/%d/%y'), datetime.strptime('12/11/18', '%m/%d/%y')],
		[6,  datetime.strptime('12/11/18', '%m/%d/%y'), datetime.strptime('01/07/19', '%m/%d/%y')],
		[7,  datetime.strptime('01/07/19', '%m/%d/%y'), datetime.strptime('02/02/19', '%m/%d/%y')],
		[8,  datetime.strptime('02/02/19', '%m/%d/%y'), datetime.strptime('02/28/19', '%m/%d/%y')],
		[9,  datetime.strptime('02/28/19', '%m/%d/%y'), datetime.strptime('03/26/19', '%m/%d/%y')],
		[10, datetime.strptime('03/26/19', '%m/%d/%y'), datetime.strptime('04/22/19', '%m/%d/%y')],
		[11, datetime.strptime('04/22/19', '%m/%d/%y'), datetime.strptime('05/21/19', '%m/%d/%y')],
		[12, datetime.strptime('05/21/19', '%m/%d/%y'), datetime.strptime('06/19/19', '%m/%d/%y')],
		[13, datetime.strptime('06/19/19', '%m/%d/%y'), datetime.strptime('07/18/19', '%m/%d/%y')],
		[14, datetime.strptime('07/18/19', '%m/%d/%y'), datetime.strptime('08/15/19', '%m/%d/%y')],
		[15, datetime.strptime('08/15/19', '%m/%d/%y'), datetime.strptime('09/11/19', '%m/%d/%y')],
		[16, datetime.strptime('09/11/19', '%m/%d/%y'), datetime.strptime('10/07/19', '%m/%d/%y')],
		[17, datetime.strptime('10/07/19', '%m/%d/%y'), datetime.strptime('11/02/19', '%m/%d/%y')],
		[18, datetime.strptime('11/02/19', '%m/%d/%y'), datetime.strptime('11/27/19', '%m/%d/%y')],
		[19, datetime.strptime('11/27/19', '%m/%d/%y'), datetime.strptime('12/24/19', '%m/%d/%y')],
		[20, datetime.strptime('12/24/19', '%m/%d/%y'), datetime.strptime('01/21/20', '%m/%d/%y')],
		[21, datetime.strptime('01/21/20', '%m/%d/%y'), datetime.strptime('02/18/20', '%m/%d/%y')],
		[22, datetime.strptime('02/18/20', '%m/%d/%y'), datetime.strptime('03/18/20', '%m/%d/%y')],
		[23, datetime.strptime('03/18/20', '%m/%d/%y'), datetime.strptime('04/16/20', '%m/%d/%y')],
		[24, datetime.strptime('04/16/20', '%m/%d/%y'), datetime.strptime('05/13/20', '%m/%d/%y')],
		[25, datetime.strptime('05/13/20', '%m/%d/%y'), datetime.strptime('06/08/20', '%m/%d/%y')],
		[26, datetime.strptime('06/08/20', '%m/%d/%y'), datetime.strptime('07/04/20', '%m/%d/%y')],
		[27, datetime.strptime('07/04/20', '%m/%d/%y'), datetime.strptime('07/30/20', '%m/%d/%y')],
		[28, datetime.strptime('07/30/20', '%m/%d/%y'), datetime.strptime('08/26/20', '%m/%d/%y')],
		[29, datetime.strptime('08/26/20', '%m/%d/%y'), datetime.strptime('09/22/20', '%m/%d/%y')],
		[30, datetime.strptime('09/22/20', '%m/%d/%y'), datetime.strptime('10/21/20', '%m/%d/%y')],
		[31, datetime.strptime('10/21/20', '%m/%d/%y'), datetime.strptime('11/19/20', '%m/%d/%y')],
		[32, datetime.strptime('11/19/20', '%m/%d/%y'), datetime.strptime('12/17/20', '%m/%d/%y')],
		[33, datetime.strptime('12/17/20', '%m/%d/%y'), datetime.strptime('01/13/21', '%m/%d/%y')],
		[34, datetime.strptime('01/13/21', '%m/%d/%y'), datetime.strptime('02/09/21', '%m/%d/%y')],
		[35, datetime.strptime('02/09/21', '%m/%d/%y'), datetime.strptime('03/07/21', '%m/%d/%y')],
		[36, datetime.strptime('03/07/21', '%m/%d/%y'), datetime.strptime('04/02/21', '%m/%d/%y')],
		[37, datetime.strptime('04/02/21', '%m/%d/%y'), datetime.strptime('04/28/21', '%m/%d/%y')],
		[38, datetime.strptime('04/28/21', '%m/%d/%y'), datetime.strptime('05/26/21', '%m/%d/%y')],
		[39, datetime.strptime('05/26/21', '%m/%d/%y'), datetime.strptime('06/24/21', '%m/%d/%y')],
	]

	fig = plt.figure(figsize=(20,6))
	ax = fig.add_subplot(111)

	for k, fpath in enumerate(tqdm(files)):
		fname = os.path.basename(fpath)

		# This requires the "brief" utility tool
		# https://naif.jpl.nasa.gov/naif/utilities.html
		# TODO: There is proberly a way to do this with SpiceyPy
		proc = subprocess.Popen('brief -s --95 "' + fpath + '"', shell=True, stdout=subprocess.PIPE, universal_newlines=True)
		stdout, stderr = proc.communicate()
		lines = stdout.split("\n")

		# Parse the two dates into datetime objects:
		date_line = lines[9].strip()
		m = re.match('^(\d{4} .{3} \d{2} \d{2}:\d{2}:\d{2})\.(\d+)\s+(\d{4} .{3} \d{2} \d{2}:\d{2}:\d{2})\.(\d+)$', date_line)
		if m is None:
			print(stdout)
			continue

		dt_start = datetime.strptime(m.group(1) + '.{:0<6}'.format(m.group(2)), "%Y %b %d %H:%M:%S.%f")
		dt_end = datetime.strptime(m.group(3) + '.{:0<6}'.format(m.group(4)), "%Y %b %d %H:%M:%S.%f")

		# Plot in different style depending on the type of kernel:
		if 'DEF' in fname:
			ax.plot([dt_start, dt_end], [k, k], 'b.-', alpha=0.5, label=fname)
			ax.fill_between([dt_start, dt_end], 0, k, color='b', alpha=0.1
					   )
		elif 'PRE_LONG' in fname:
			ax.plot([dt_start, dt_end], [k, k], 'r.-', alpha=0.5)
			ax.fill_between([dt_start, dt_end], 0, k, color='r', alpha=0.1)

		elif 'PRE_COMM' in fname:
			ax.plot([dt_start, dt_end], [k, k], 'g.-', alpha=0.5)
			ax.fill_between([dt_start, dt_end], 0, k, color='g', alpha=0.1)

		elif 'PRE_MNVR' in fname:
			ax.plot([dt_start, dt_end], [k, k], 'c.-', alpha=0.5)
			ax.fill_between([dt_start, dt_end], 0, k, color='c', alpha=0.1)

		elif 'PRE' in fname:
			ax.plot([dt_start, dt_end], [k, k], 'y.-', alpha=0.5)
			ax.fill_between([dt_start, dt_end], 0, k, color='y', alpha=0.1)

		else:
			print(fname)

	# Plot sectors as well:
	for s in sectors:
		ax.axvline(s[1], color='0.7', ls=':')
		ax.text(s[1] + (s[2] - s[1])/2, k+10, '%d' % s[0], horizontalalignment='center', verticalalignment='top')

	ax.set_ylim(bottom=0)
	ax.set_ylabel('Kernel number')
	ax.set_xlabel('Time (ET)')
	plt.show()
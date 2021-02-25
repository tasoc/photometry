#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the time coverage of the loaded SPICE kernels.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import subprocess
import re
from datetime import datetime
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.spice import TESS_SPICE

if __name__ == '__main__':
	# Switch back to interactive plotting backend:
	# Importing photometry currently changes the backend to Agg, which is non-interactive.
	plt.switch_backend('Qt5Agg')

	# List of sectors from tess.mit.edu:
	# TODO: Are these actually in UTC?
	sectors = [
		[0,  datetime.strptime('04/18/18 22:51:00.338000 UTC', '%m/%d/%y %H:%M:%S.%f %Z'), datetime.strptime('07/25/18', '%m/%d/%y')],
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
		[40, datetime.strptime('06/24/21', '%m/%d/%y'), datetime.strptime('07/23/21', '%m/%d/%y')],
		[41, datetime.strptime('07/23/21', '%m/%d/%y'), datetime.strptime('08/20/21', '%m/%d/%y')],
		[42, datetime.strptime('08/20/21', '%m/%d/%y'), datetime.strptime('09/16/21', '%m/%d/%y')],
		[43, datetime.strptime('09/16/21', '%m/%d/%y'), datetime.strptime('10/12/21', '%m/%d/%y')],
		[44, datetime.strptime('10/12/21', '%m/%d/%y'), datetime.strptime('11/06/21', '%m/%d/%y')],
		[45, datetime.strptime('11/06/21', '%m/%d/%y'), datetime.strptime('12/02/21', '%m/%d/%y')],
		[46, datetime.strptime('12/02/21', '%m/%d/%y'), datetime.strptime('12/30/21', '%m/%d/%y')],
		[47, datetime.strptime('12/30/21', '%m/%d/%y'), datetime.strptime('01/28/22', '%m/%d/%y')],
		[48, datetime.strptime('01/28/22', '%m/%d/%y'), datetime.strptime('02/26/22', '%m/%d/%y')],
		[49, datetime.strptime('02/26/22', '%m/%d/%y'), datetime.strptime('03/26/22', '%m/%d/%y')],
		[50, datetime.strptime('03/26/22', '%m/%d/%y'), datetime.strptime('04/22/22', '%m/%d/%y')],
		[51, datetime.strptime('04/22/22', '%m/%d/%y'), datetime.strptime('05/18/22', '%m/%d/%y')],
		[52, datetime.strptime('05/18/22', '%m/%d/%y'), datetime.strptime('06/13/22', '%m/%d/%y')],
		[53, datetime.strptime('06/13/22', '%m/%d/%y'), datetime.strptime('07/09/22', '%m/%d/%y')],
		[54, datetime.strptime('07/09/22', '%m/%d/%y'), datetime.strptime('08/05/22', '%m/%d/%y')],
		[55, datetime.strptime('08/05/22', '%m/%d/%y'), datetime.strptime('09/01/22', '%m/%d/%y')],
	]

	fig = plt.figure(figsize=(15,6), dpi=100)
	ax = fig.add_subplot(111)

	with TESS_SPICE() as ts:
		# This requires the "brief" utility tool
		# https://naif.jpl.nasa.gov/naif/utilities.html
		# TODO: There is properly a way to do this with SpiceyPy
		proc = subprocess.Popen('brief --95 -utc "' + ts.METAKERNEL + '"', shell=True, stdout=subprocess.PIPE, universal_newlines=True)
		stdout, stderr = proc.communicate()
		lines = stdout.split("\n")

		k = -1
		for line in lines:
			line = line.strip()
			if not line: continue
			if line.startswith('Summary for: '):
				fpath = line[13:]
				fname = os.path.basename(fpath)
				k += 1
				continue

			m = re.match(r'^(\d{4}-.{3}-\d{2} \d{2}:\d{2}:\d{2})\.(\d+)\s+(\d{4}-.{3}-\d{2} \d{2}:\d{2}:\d{2})\.(\d+)$', line)
			if m is not None:
				dt_start = datetime.strptime(m.group(1) + '.{:0<6}'.format(m.group(2)), "%Y-%b-%d %H:%M:%S.%f")
				dt_end = datetime.strptime(m.group(3) + '.{:0<6}'.format(m.group(4)), "%Y-%b-%d %H:%M:%S.%f")

				print(fname + ": " + str(dt_start) + " - " + str(dt_end))

				# Plot in different style depending on the type of kernel:
				if 'DEF' in fname:
					color = 'b'
				elif 'PRE_LONG' in fname:
					color = 'r'
				elif 'PRE_MNVR' in fname:
					color = 'g'
				elif 'PRE_COMM' in fname:
					color = 'c'
				elif 'PRE' in fname:
					color = 'y'
				else:
					print("???????")

				ax.plot([dt_start, dt_end], [k, k], color=color, ls='-', marker='.', alpha=0.5, label=fname)
				ax.fill_between([dt_start, dt_end], 0, k, color=color, alpha=0.1)

				fpath = None
				fname = None

		ts.unload()

	# Plot sectors as well:
	for s in sectors:
		ax.axvline(s[1], color='0.7', ls=':')
		ax.text(s[1] + (s[2] - s[1])/2, k+10, '%d' % s[0], horizontalalignment='center', verticalalignment='top')
	ax.axvline(sectors[-1][2], color='0.7', ls=':')

	ax.set_xlim(left=datetime(2018, 3, 20))
	ax.set_ylim(bottom=0)
	# For zooming in on pre-sector1:
	#ax.set_xlim(right=datetime(2018, 8, 15))
	#ax.set_ylim(top=45)

	ax.set_ylabel('Kernel number (importance)')
	ax.set_xlabel('Time (UTC)')
	fig.savefig('spice_coverage.png', bbox_inches='tight')

	plt.show()

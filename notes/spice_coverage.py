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
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.spice import TESS_SPICE
from photometry.plots import plt, plots_interactive

if __name__ == '__main__':
	# Switch back to interactive plotting backend:
	# Importing photometry currently changes the backend to Agg, which is non-interactive.
	plots_interactive()

	# List of sectors from tess.mit.edu:
	# TODO: Are these actually in UTC?
	sectors = [
		[0, datetime.strptime('04/18/18 22:51:00.338000 UTC', '%m/%d/%y %H:%M:%S.%f %Z'), '07/25/18'],
		[1, '07/25/18', '08/22/18'],
		[2, '08/22/18', '09/20/18'],
		[3, '09/20/18', '10/18/18'],
		[4, '10/18/18', '11/15/18'],
		[5, '11/15/18', '12/11/18'],
		[6, '12/11/18', '01/07/19'],
		[7, '01/07/19', '02/02/19'],
		[8, '02/02/19', '02/28/19'],
		[9, '02/28/19', '03/26/19'],
		[10, '03/26/19', '04/22/19'],
		[11, '04/22/19', '05/21/19'],
		[12, '05/21/19', '06/19/19'],
		[13, '06/19/19', '07/18/19'],
		[14, '07/18/19', '08/15/19'],
		[15, '08/15/19', '09/11/19'],
		[16, '09/11/19', '10/07/19'],
		[17, '10/07/19', '11/02/19'],
		[18, '11/02/19', '11/27/19'],
		[19, '11/27/19', '12/24/19'],
		[20, '12/24/19', '01/21/20'],
		[21, '01/21/20', '02/18/20'],
		[22, '02/18/20', '03/18/20'],
		[23, '03/18/20', '04/16/20'],
		[24, '04/16/20', '05/13/20'],
		[25, '05/13/20', '06/08/20'],
		[26, '06/08/20', '07/04/20'],
		[27, '07/04/20', '07/30/20'],
		[28, '07/30/20', '08/26/20'],
		[29, '08/26/20', '09/22/20'],
		[30, '09/22/20', '10/21/20'],
		[31, '10/21/20', '11/19/20'],
		[32, '11/19/20', '12/17/20'],
		[33, '12/17/20', '01/13/21'],
		[34, '01/13/21', '02/09/21'],
		[35, '02/09/21', '03/07/21'],
		[36, '03/07/21', '04/02/21'],
		[37, '04/02/21', '04/28/21'],
		[38, '04/28/21', '05/26/21'],
		[39, '05/26/21', '06/24/21'],
		[40, '06/24/21', '07/23/21'],
		[41, '07/23/21', '08/20/21'],
		[42, '08/20/21', '09/16/21'],
		[43, '09/16/21', '10/12/21'],
		[44, '10/12/21', '11/06/21'],
		[45, '11/06/21', '12/02/21'],
		[46, '12/02/21', '12/30/21'],
		[47, '12/30/21', '01/28/22'],
		[48, '01/28/22', '02/26/22'],
		[49, '02/26/22', '03/26/22'],
		[50, '03/26/22', '04/22/22'],
		[51, '04/22/22', '05/18/22'],
		[52, '05/18/22', '06/13/22'],
		[53, '06/13/22', '07/09/22'],
		[54, '07/09/22', '08/05/22'],
		[55, '08/05/22', '09/01/22'],
		[56, '09/01/22', '09/30/22'],
		[57, '09/30/22', '10/29/22'],
		[58, '10/29/22', '11/26/22'],
		[59, '11/26/22', '12/23/22'],
		[60, '12/23/22', '01/18/23'],
		[61, '01/18/23', '02/12/23'],
		[62, '02/12/23', '03/10/23'],
		[63, '03/10/23', '04/06/23'],
		[64, '04/06/23', '05/04/23'],
		[65, '05/04/23', '06/02/23'],
		[66, '06/02/23', '07/01/23'],
		[67, '07/01/23', '07/29/23'],
		[68, '07/29/23', '08/25/23'],
		[69, '08/25/23', '09/20/23'],
		# Year 6:
		[70, '09/20/23', '10/16/23'],
		[71, '10/16/23', '11/11/23'],
		[72, '11/11/23', '12/07/23'],
		[73, '12/07/23', '01/03/24'],
		[74, '01/03/24', '01/30/24'],
		[75, '01/30/24', '02/26/24'],
		[76, '02/26/24', '03/26/24'],
		[77, '03/26/24', '04/23/24'],
		[78, '04/23/24', '05/21/24'],
		[79, '05/21/24', '06/18/24'],
		[80, '06/18/24', '07/15/24'],
		[81, '07/15/24', '08/10/24'],
		[82, '08/10/24', '09/05/24'],
		[83, '09/05/24', '10/01/24'],
	]

	for k in range(len(sectors)):
		if isinstance(sectors[k][1], str):
			sectors[k][1] = datetime.strptime(sectors[k][1], '%m/%d/%y')
		if isinstance(sectors[k][2], str):
			sectors[k][2] = datetime.strptime(sectors[k][2], '%m/%d/%y')

	fig = plt.figure(figsize=(15,6), dpi=100)
	ax = fig.add_subplot(111)

	with TESS_SPICE(download=True) as ts:
		for k, row in enumerate(ts.time_coverage('TESS')):

			fname, tmin, tmax = row
			print(fname + "," + tmin.utc.iso + "," + tmax.utc.iso)

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

			ax.plot([tmin.datetime, tmax.datetime], [k, k], color=color, ls='-', marker='.', alpha=0.5, label=fname)
			ax.fill_between([tmin.datetime, tmax.datetime], 0, k, color=color, alpha=0.1)

		"""
		print('-'*70)

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

				print(fname + "," + str(dt_start) + "," + str(dt_end))

				fpath = None
				fname = None
		"""

		ts.unload()

	# Plot sectors as well:
	for s in sectors:
		ax.axvline(s[1], color='0.7', ls=':')
		ax.text(s[1] + (s[2] - s[1])/2, k+10, '%d' % s[0], horizontalalignment='center', verticalalignment='top')
	ax.axvline(sectors[-1][2], color='0.7', ls=':')

	# Plot now:
	ax.axvline(datetime.utcnow(), color='r', ls='--')

	ax.set_xlim(left=datetime(2018, 3, 20))
	ax.set_ylim(bottom=0)
	# For zooming in on pre-sector1:
	#ax.set_xlim(right=datetime(2018, 8, 15))
	#ax.set_ylim(top=45)

	ax.set_ylabel('Kernel number (importance)')
	ax.set_xlabel('Time (UTC)')
	fig.savefig('spice_coverage.png', bbox_inches='tight')

	plt.show()

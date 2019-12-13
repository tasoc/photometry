#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A TaskManager which keeps track of which targets to process.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import os
import sqlite3
import logging
import json
from . import STATUS

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, overwrite=False, summary=None, summary_interval=100):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file (string): Path to the TODO-file.
			cleanup (boolean): Perform cleanup/optimization of TODO-file before
				during initialization. Default=False.
			overwrite (boolean): Restart calculation from the beginning, discarding any previous results. Default=False.
			summary (string): Path to file where to periodically write a progress summary. The output file will be in JSON format. Default=None.
			summary_interval (int): Interval at which to write summary file. Setting this to 1 will mean writing the file after every tasks completes. Default=100.

		Raises:
			FileNotFoundError: If TODO-file could not be found.
		"""

		self.overwrite = overwrite
		self.summary_file = summary
		self.summary_interval = summary_interval

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.exists(todo_file):
			raise FileNotFoundError('Could not find TODO-file')

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		self.logger.addHandler(console)
		self.logger.setLevel(logging.INFO)

		# Load the SQLite file:
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()
		self.cursor.execute("PRAGMA foreign_keys=ON;")
		self.cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")
		self.cursor.execute("PRAGMA journal_mode=TRUNCATE;")

		# Reset the status of everything for a new run:
		if overwrite:
			self.cursor.execute("UPDATE todolist SET status=NULL;")
			self.cursor.execute("DROP TABLE IF EXISTS diagnostics;")
			self.cursor.execute("DROP TABLE IF EXISTS photometry_skipped;")
			self.conn.commit()

		# Create table for diagnostics:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS diagnostics (
			priority INTEGER PRIMARY KEY ASC NOT NULL,
			starid BIGINT NOT NULL,
			lightcurve TEXT,
			elaptime REAL NOT NULL,
			mean_flux DOUBLE PRECISION,
			variance DOUBLE PRECISION,
			variability DOUBLE PRECISION,
			rms_hour DOUBLE PRECISION,
			ptp DOUBLE PRECISION,
			pos_row REAL,
			pos_column REAL,
			contamination REAL,
			mask_size INT,
			edge_flux REAL,
			stamp_width INT,
			stamp_height INT,
			stamp_resizes INT,
			errors TEXT,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS photometry_skipped (
			priority INTEGER NOT NULL,
			skipped_by INTEGER NOT NULL,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE,
			FOREIGN KEY (skipped_by) REFERENCES todolist(priority) ON DELETE RESTRICT ON UPDATE CASCADE
		);""") # PRIMARY KEY
		self.conn.commit()

		# Add status indicator for corrections to todolist, if it doesn't already exists:
		self.cursor.execute("PRAGMA table_info(diagnostics)")
		if 'edge_flux' not in [r['name'] for r in self.cursor.fetchall()]:
			self.logger.debug("Adding edge_flux column to diagnostics")
			self.cursor.execute("ALTER TABLE diagnostics ADD COLUMN edge_flux REAL DEFAULT NULL")
			self.conn.commit()

		# Reset calculations with status STARTED, ABORT or ERROR:
		# We are re-running all with error, in the hope that they will work this time around:
		clear_status = str(STATUS.STARTED.value) + ',' + str(STATUS.ABORT.value) + ',' + str(STATUS.ERROR.value)
		self.cursor.execute("UPDATE todolist SET status=NULL WHERE status IN (" + clear_status + ");")
		self.conn.commit()

		# Analyze the tables for better query planning:
		self.cursor.execute("ANALYZE;")

		# Prepare summary object:
		self.summary = {
			'slurm_jobid': os.environ.get('SLURM_JOB_ID', None),
			'numtasks': 0,
			'tasks_run': 0,
			'last_error': None,
			'mean_elaptime': None
		}
		# Make sure to add all the different status to summary:
		for s in STATUS: self.summary[s.name] = 0
		# If we are going to output summary, make sure to fill it up:
		if self.summary_file:
			# Extract information from database:
			self.cursor.execute("SELECT status,COUNT(*) AS cnt FROM todolist GROUP BY status;")
			for row in self.cursor.fetchall():
				self.summary['numtasks'] += row['cnt']
				if row['status'] is not None:
					self.summary[STATUS(row['status']).name] = row['cnt']
			# Write summary to file:
			self.write_summary()

		# Run a cleanup/optimization of the database before we get started:
		if cleanup:
			self.logger.info("Cleaning TODOLIST before run...")
			try:
				self.conn.isolation_level = None
				self.cursor.execute("VACUUM;")
			except:
				raise
			finally:
				self.conn.isolation_level = ''

	def close(self):
		"""Close TaskManager and all associated objects."""
		self.cursor.close()
		self.conn.close()
		self.write_summary()

	def __exit__(self, *args):
		self.close()

	def __enter__(self):
		return self

	def get_number_tasks(self):
		"""
		Get number of tasks due to be processed.

		Returns:
			int: Number of tasks due to be processed.
		"""
		self.cursor.execute("SELECT COUNT(*) AS num FROM todolist WHERE status IS NULL;")
		num = int(self.cursor.fetchone()['num'])
		return num

	def get_task(self, starid=None):
		"""
		Get next task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""
		constraints = []
		if starid is not None:
			constraints.append("starid=%d" % starid)

		if constraints:
			constraints = " AND " + " AND ".join(constraints)
		else:
			constraints = ''

		self.cursor.execute("SELECT priority,starid,method,sector,camera,ccd,datasource,tmag FROM todolist WHERE status IS NULL" + constraints + " ORDER BY priority LIMIT 1;")
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	def get_random_task(self):
		"""
		Get random task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""
		self.cursor.execute("SELECT priority,starid,method,sector,camera,ccd,datasource,tmag FROM todolist WHERE status IS NULL ORDER BY RANDOM() LIMIT 1;")
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	def save_result(self, result):
		"""
		Save results and diagnostics. This will update the TODO list.

		Parameters:
			results (dict): Dictionary of results and diagnostics.
		"""

		# Extract details dictionary:
		details = result.get('details', {})

		# The status of this target returned by the photometry:
		my_status = result['status']

		# Also set status of targets that were marked as "SKIPPED" by this target:
		if 'skip_targets' in details and len(details['skip_targets']) > 0:
			skip_targets = set(details['skip_targets'])
			if result['datasource'].startswith('tpf:') and int(result['datasource'][4:]) in skip_targets:
				# This secondary target is in the mask of the primary target.
				# We never want to return a lightcurve for a secondary target over
				# a primary target, so we are going to mark this one as SKIPPED.
				self.logger.info("Changing status to SKIPPED for priority %s because it overlaps with primary target", result['priority'])
				my_status = STATUS.SKIPPED
			else:
				# Create unique list of starids to be masked as skipped:
				skip_starids = [str(starid) for starid in skip_targets]
				skip_starids = ','.join(skip_starids)

				# Ask the todolist if there are any stars that are brighter than this
				# one among the other targets in the mask:
				if result['datasource'] == 'tpf':
					skip_datasources = "'tpf','tpf:%d'" % result['starid']
				else:
					skip_datasources = "'" + result['datasource'] + "'"

				self.cursor.execute("SELECT priority,tmag FROM todolist WHERE starid IN (" + skip_starids + ") AND datasource IN (" + skip_datasources + ") AND sector=?;", (result['sector'],))
				skip_rows = self.cursor.fetchall()
				if len(skip_rows) > 0:
					skip_tmags = np.array([row['tmag'] for row in skip_rows])
					if np.all(result['tmag'] < skip_tmags):
						# This target was the brightest star in the mask,
						# so let's keep it and simply mark all the other targets
						# as SKIPPED:
						self.cursor.execute("DELETE FROM photometry_skipped WHERE skipped_by=?;", (result['priority'],))
						for row in skip_rows:
							self.cursor.execute("UPDATE todolist SET status=? WHERE priority=?;", (
								STATUS.SKIPPED.value,
								row['priority']
							))
							self.summary['SKIPPED'] += self.cursor.rowcount
							self.cursor.execute("INSERT INTO photometry_skipped (priority,skipped_by) VALUES (?,?);", (
								row['priority'],
								result['priority']
							))
					else:
						# This target was not the brightest star in the mask,
						# and a brighter target is going to be processed,
						# so let's change this one to SKIPPED and let the other
						# one run later on
						self.logger.info("Changing status to SKIPPED for priority %s", result['priority'])
						my_status = STATUS.SKIPPED

		# Update the status in the TODO list:
		self.cursor.execute("UPDATE todolist SET status=? WHERE priority=?;", (
			my_status.value,
			result['priority']
		))
		self.summary['tasks_run'] += 1
		self.summary[my_status.name] += 1
		self.summary['STARTED'] -= 1

		# Save additional diagnostics:
		error_msg = details.get('errors', None)
		if error_msg:
			error_msg = '\n'.join(error_msg)
			self.summary['last_error'] = error_msg

		# Calculate mean elapsed time using "streaming weighted mean" with (alpha=0.1):
		# https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl
		if self.summary['mean_elaptime'] is None:
			self.summary['mean_elaptime'] = result['time']
		else:
			self.summary['mean_elaptime'] += 0.1 * (result['time'] - self.summary['mean_elaptime'])

		stamp = details.get('stamp', None)
		stamp_width = None if stamp is None else stamp[3] - stamp[2]
		stamp_height = None if stamp is None else stamp[1] - stamp[0]

		self.cursor.execute("INSERT OR REPLACE INTO diagnostics (priority, starid, lightcurve, elaptime, pos_column, pos_row, mean_flux, variance, variability, rms_hour, ptp, mask_size, edge_flux, contamination, stamp_width, stamp_height, stamp_resizes, errors) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", (
			result['priority'],
			result['starid'],
			details.get('filepath_lightcurve', None),
			result['time'],
			details.get('pos_centroid', (None, None))[0],
			details.get('pos_centroid', (None, None))[1],
			details.get('mean_flux', None),
			details.get('variance', None),
			details.get('variability', None),
			details.get('rms_hour', None),
			details.get('ptp', None),
			details.get('mask_size', None),
			details.get('edge_flux', None),
			details.get('contamination', None),
			stamp_width,
			stamp_height,
			details.get('stamp_resizes', 0),
			error_msg
		))
		self.conn.commit()

		# Write summary file:
		if self.summary_file and self.summary['tasks_run'] % self.summary_interval == 0:
			self.write_summary()

	def start_task(self, taskid):
		"""
		Mark a task as STARTED in the TODO-list.
		"""
		self.cursor.execute("UPDATE todolist SET status=? WHERE priority=?;", (STATUS.STARTED.value, taskid))
		self.conn.commit()
		self.summary['STARTED'] += 1

	def write_summary(self):
		"""Write summary of progress to file. The summary file will be in JSON format."""
		if self.summary_file:
			try:
				with open(self.summary_file, 'w') as fid:
					json.dump(self.summary, fid)
			except:
				self.logger.exception("Could not write summary file")

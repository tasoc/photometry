#!/usr/bin/env python3
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
from numpy import atleast_1d
from . import STATUS, utilities

#--------------------------------------------------------------------------------------------------
def build_constraints(priority=None, starid=None, sector=None, cadence=None,
	camera=None, ccd=None, cbv_area=None, datasource=None, tmag_min=None, tmag_max=None):
	"""
	Build constraints for database query from given parameters.

	For ``tmag_min`` and ``tmag_max`` constraints, these limits are put on the primary target
	for all secondary targets. This means that a faint target will still be processed if it is
	in the TPF of a bright target. This is because this constraint is primarily used for
	processing bright targets separately since these require more memory.

	Parameters:
		priority (int, optional): Only return task matching this priority.
		starid (int, optional): Only return tasks matching this starid.
		sector (int, optional): Only return tasks matching this Sector.
		cadence (int, optional): Only return tasks matching this cadence.
		camera (int, optional): Only return tasks matching this camera.
		ccd (int, optional): Only return tasks matching this CCD.
		cbv_area (int, optional): Only return tasks matching this CBV-AREA.
		datasource (str, optional): Only return tasks from this datasource.
			Choises are ``'tpf'`` and ``'ffi'``.
		tmag_min (float, optional): Lower/bright limit on Tmag.
		tmag_max (float, optional): Upper/faint limit on Tmag.

	Returns:
		list: List of strings containing constraints for database. The constraints should be
			joined with "AND" to have the desired effect.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	constraints = []
	if priority is not None:
		constraints.append('todolist.priority IN (' + ','.join([str(int(c)) for c in atleast_1d(priority)]) + ')')
	if starid is not None:
		constraints.append('todolist.starid IN (' + ','.join([str(int(c)) for c in atleast_1d(starid)]) + ')')
	if sector is not None:
		constraints.append('todolist.sector IN (' + ','.join([str(int(c)) for c in atleast_1d(sector)]) + ')')
	if cadence == 'ffi':
		constraints.append("todolist.datasource='ffi'")
	elif cadence is not None:
		constraints.append('todolist.cadence IN (' + ','.join([str(int(c)) for c in atleast_1d(cadence)]) + ')')
	if camera is not None:
		constraints.append('todolist.camera IN (' + ','.join([str(int(c)) for c in atleast_1d(camera)]) + ')')
	if ccd is not None:
		constraints.append('todolist.ccd IN (' + ','.join([str(int(c)) for c in atleast_1d(ccd)]) + ')')
	if cbv_area is not None:
		constraints.append('todolist.cbv_area IN (' + ','.join([str(int(c)) for c in atleast_1d(cbv_area)]) + ')')

	if tmag_min is not None or tmag_max is not None:
		# To avoid having three separate cases, we join all cases by
		# putting in dummy upper and lower bounds in case they are
		# not provided. The values should be outside the range on any normal stars:
		tmag_min = -99 if tmag_min is None else tmag_min
		tmag_max = 99 if tmag_max is None else tmag_max
		constraints.append(f"((todolist.datasource NOT LIKE 'tpf:%' AND todolist.tmag BETWEEN {tmag_min:f} AND {tmag_max:f}) OR (todolist.datasource LIKE 'tpf:%' AND CAST(SUBSTR(todolist.datasource,5) AS INTEGER) IN (SELECT DISTINCT starid FROM todolist t2 WHERE t2.datasource='tpf' AND t2.tmag BETWEEN {tmag_min:f} AND {tmag_max:f})))")

	if datasource is not None:
		constraints.append("todolist.datasource='ffi'" if datasource == 'ffi' else "todolist.datasource!='ffi'")

	return constraints

#--------------------------------------------------------------------------------------------------
class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file, cleanup=False, overwrite=False, cleanup_constraints=None,
		summary=None, summary_interval=100):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file (string): Path to the TODO-file.
			cleanup (boolean, optional): Perform cleanup/optimization of TODO-file before
				during initialization. Default=False.
			overwrite (boolean, optional): Restart calculation from the beginning, discarding
				any previous results. Default=False.
			cleanup_constraints (dict, optional): Dict of constraint for cleanup of the status of
				previous correction runs. If not specified, all bad results are cleaned up.
			summary (string, optional): Path to file where to periodically write a progress summary.
				The output file will be in JSON format. Default=None.
			summary_interval (int, optional): Interval at which summary file is updated.
				Setting this to 1 will mean writing the file after every tasks completes.
				Default=100.

		Raises:
			FileNotFoundError: If TODO-file could not be found.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		self.overwrite = overwrite
		self.summary_file = summary
		self.summary_interval = summary_interval
		self.summary_counter = 0

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.exists(todo_file):
			raise FileNotFoundError('Could not find TODO-file')

		if cleanup_constraints is not None and not isinstance(cleanup_constraints, (dict, list)):
			raise ValueError("cleanup_constraints should be dict or list")

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		if not self.logger.hasHandlers():
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
			lightcurve TEXT,
			method_used TEXT NOT NULL,
			elaptime REAL NOT NULL,
			worker_wait_time REAL,
			mean_flux DOUBLE PRECISION,
			variance DOUBLE PRECISION,
			variability DOUBLE PRECISION,
			rms_hour DOUBLE PRECISION,
			ptp DOUBLE PRECISION,
			pos_row REAL,
			pos_column REAL,
			contamination REAL,
			mask_size INTEGER,
			edge_flux REAL,
			stamp_width INTEGER,
			stamp_height INTEGER,
			stamp_resizes INTEGER,
			errors TEXT,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE
		);""")
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS photometry_skipped (
			priority INTEGER NOT NULL,
			skipped_by INTEGER NOT NULL,
			FOREIGN KEY (priority) REFERENCES todolist(priority) ON DELETE CASCADE ON UPDATE CASCADE,
			FOREIGN KEY (skipped_by) REFERENCES todolist(priority) ON DELETE RESTRICT ON UPDATE CASCADE
		);""")
		self.cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS diagnostics_lightcurve_idx ON diagnostics (lightcurve);")
		self.cursor.execute("CREATE INDEX IF NOT EXISTS todolist_datasource_idx ON todolist (datasource);")
		self.conn.commit()

		# This is only for backwards compatibility.
		self.cursor.execute("PRAGMA table_info(todolist)")
		existing_columns = [r['name'] for r in self.cursor.fetchall()]
		if 'cadence' not in existing_columns:
			self.logger.debug("Adding CADENCE column to todolist")
			self.cursor.execute("BEGIN TRANSACTION;")
			self.cursor.execute("ALTER TABLE todolist ADD COLUMN cadence INTEGER DEFAULT NULL;")
			self.cursor.execute("UPDATE todolist SET cadence=1800 WHERE datasource='ffi' AND sector < 27;")
			self.cursor.execute("UPDATE todolist SET cadence=600 WHERE datasource='ffi' AND sector >= 27 AND sector <= 55;")
			self.cursor.execute("UPDATE todolist SET cadence=120 WHERE datasource!='ffi' AND sector < 27;")
			self.cursor.execute("SELECT COUNT(*) AS antal FROM todolist WHERE cadence IS NULL;")
			if self.cursor.fetchone()['antal'] > 0:
				self.close()
				raise ValueError("TODO-file does not contain CADENCE information and it could not be determined automatically. Please recreate TODO-file.")
			self.conn.commit()

		# Add status indicator for corrections to todolist, if it doesn't already exists:
		# This is only for backwards compatibility.
		self.cursor.execute("PRAGMA table_info(diagnostics)")
		existing_columns = [r['name'] for r in self.cursor.fetchall()]
		if 'edge_flux' not in existing_columns:
			self.logger.debug("Adding edge_flux column to diagnostics")
			self.cursor.execute("ALTER TABLE diagnostics ADD COLUMN edge_flux REAL DEFAULT NULL")
			self.conn.commit()
		if 'worker_wait_time' not in existing_columns:
			self.logger.debug("Adding worker_wait_time column to diagnostics")
			self.cursor.execute("ALTER TABLE diagnostics ADD COLUMN worker_wait_time REAL DEFAULT NULL")
			self.conn.commit()
		if 'method_used' not in existing_columns:
			# Since this one is NOT NULL, we have to do some magic to fill out the
			# new column after creation, by finding keywords in other columns.
			# This can be a pretty slow process, but it only has to be done once.
			self.logger.debug("Adding method_used column to diagnostics")
			self.cursor.execute("BEGIN TRANSACTION;")
			self.cursor.execute("ALTER TABLE diagnostics ADD COLUMN method_used TEXT NOT NULL DEFAULT 'aperture';")
			for m in ('aperture', 'halo', 'psf', 'linpsf'):
				self.cursor.execute("UPDATE diagnostics SET method_used=? WHERE priority IN (SELECT priority FROM todolist WHERE method=?);", [m, m])
			self.cursor.execute("UPDATE diagnostics SET method_used='halo' WHERE method_used='aperture' AND errors LIKE '%Automatically switched to Halo photometry%';")
			self.conn.commit()
		if 'starid' in existing_columns:
			# Drop this column from the diagnostics table, since the information is already in
			# the todolist table. Use utility function for this, since SQLite does not
			# have a DROP COLUMN mechanism directly.
			utilities.sqlite_drop_column(self.conn, 'diagnostics', 'starid')

		# Reset calculations with status STARTED, ABORT or ERROR:
		# We are re-running all with error, in the hope that they will work this time around:
		clear_status = str(STATUS.STARTED.value) + ',' + str(STATUS.ABORT.value) + ',' + str(STATUS.ERROR.value)
		constraints = ['status IN (' + clear_status + ')']

		# Add additional constraints from the user input and build SQL query:
		if cleanup_constraints:
			if isinstance(cleanup_constraints, dict):
				constraints += build_constraints(**cleanup_constraints)
			else:
				constraints += cleanup_constraints

		constraints = ' AND '.join(constraints)
		self.cursor.execute("BEGIN TRANSACTION;")
		self.cursor.execute("DELETE FROM diagnostics WHERE priority IN (SELECT todolist.priority FROM todolist WHERE " + constraints + ");")
		self.cursor.execute("UPDATE todolist SET status=NULL WHERE " + constraints + ";")
		self.conn.commit()

		# Analyze the tables for better query planning:
		self.logger.debug("Analyzing database...")
		self.cursor.execute("ANALYZE;")

		# Prepare summary object:
		self.summary = {
			'slurm_jobid': os.environ.get('SLURM_JOB_ID', None),
			'numtasks': 0,
			'tasks_run': 0,
			'last_error': None,
			'mean_elaptime': None,
			'mean_worker_waittime': None
		}
		# Make sure to add all the different status to summary:
		for s in STATUS:
			self.summary[s.name] = 0
		# If we are going to output summary, make sure to fill it up:
		if self.summary_file:
			# Ensure it is an absolute file path:
			self.summary_file = os.path.abspath(self.summary_file)
			# Extract information from database:
			self.cursor.execute("SELECT status,COUNT(*) AS cnt FROM todolist GROUP BY status;")
			for row in self.cursor.fetchall():
				self.summary['numtasks'] += row['cnt']
				if row['status'] is not None:
					self.summary[STATUS(row['status']).name] = row['cnt']
			# Make sure the containing directory exists:
			os.makedirs(os.path.dirname(self.summary_file), exist_ok=True)
			# Write summary to file:
			self.write_summary()

		# Run a cleanup/optimization of the database before we get started:
		if cleanup:
			self.logger.info("Cleaning TODOLIST before run...")
			tmp_isolevel = self.conn.isolation_level
			try:
				self.conn.isolation_level = None
				self.cursor.execute("VACUUM;")
			finally:
				self.conn.isolation_level = tmp_isolevel

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close TaskManager and all associated objects."""
		if hasattr(self, 'cursor') and hasattr(self, 'conn'):
			try:
				self.conn.rollback()
				self.cursor.execute("PRAGMA journal_mode=DELETE;")
				self.conn.commit()
				self.cursor.close()
			except sqlite3.ProgrammingError:
				pass

		if hasattr(self, 'conn'):
			self.conn.close()

		self.write_summary()

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()

	#----------------------------------------------------------------------------------------------
	def __del__(self):
		self.summary_file = None
		self.close()

	#----------------------------------------------------------------------------------------------
	def get_number_tasks(self, **kwargs):
		"""
		Get number of tasks due to be processed.

		Parameters:
			**kwarg: Keyword arguments are passed to :func:`build_constraints`.

		Returns:
			int: Number of tasks due to be processed.
		"""
		constraints = build_constraints(**kwargs)
		constraints = ' AND ' + ' AND '.join(constraints) if constraints else ''
		self.cursor.execute("SELECT COUNT(*) AS num FROM todolist WHERE status IS NULL" + constraints + ";")
		return int(self.cursor.fetchone()['num'])

	#----------------------------------------------------------------------------------------------
	def get_task(self, **kwargs):
		"""
		Get next task to be processed.

		Parameters:
			**kwarg: Keyword arguments are passed to :func:`build_constraints`.

		Returns:
			dict or None: Dictionary of settings for task.
		"""
		constraints = build_constraints(**kwargs)
		constraints = ' AND ' + ' AND '.join(constraints) if constraints else ''
		self.cursor.execute("SELECT priority,starid,method,sector,camera,ccd,cadence,datasource,tmag FROM todolist WHERE status IS NULL" + constraints + " ORDER BY priority LIMIT 1;")
		task = self.cursor.fetchone()
		if task:
			return dict(task)
		return None

	#----------------------------------------------------------------------------------------------
	def get_random_task(self):
		"""
		Get random task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""
		self.cursor.execute("SELECT priority,starid,method,sector,camera,ccd,cadence,datasource,tmag FROM todolist WHERE status IS NULL ORDER BY RANDOM() LIMIT 1;")
		task = self.cursor.fetchone()
		if task:
			return dict(task)
		return None

	#----------------------------------------------------------------------------------------------
	def start_task(self, taskid):
		"""
		Mark a task as STARTED in the TODO-list.

		Parameters:
			taskid (int): ID (priority) of the task to be marked as STARTED.
		"""
		self.cursor.execute("UPDATE todolist SET status=? WHERE priority=?;", (STATUS.STARTED.value, taskid))
		self.conn.commit()
		self.summary['STARTED'] += 1

	#----------------------------------------------------------------------------------------------
	def save_result(self, result):
		"""
		Save results and diagnostics. This will update the TODO list.

		Parameters:
			results (dict): Dictionary of results and diagnostics.
		"""

		# Extract details dictionary:
		details = result.get('details', {})
		error_msg = details.get('errors', [])

		# The status of this target returned by the photometry:
		my_status = result['status']

		# Extract stamp width and height:
		stamp = details.get('stamp', None)
		stamp_width = None if stamp is None else stamp[3] - stamp[2]
		stamp_height = None if stamp is None else stamp[1] - stamp[0]

		# Make changes to database:
		additional_skipped = 0
		self.cursor.execute("BEGIN TRANSACTION;")
		try:
			# Also set status of targets that were marked as "SKIPPED" by this target:
			if 'skip_targets' in details and len(details['skip_targets']) > 0:
				skip_targets = set(details['skip_targets'])
				if result['datasource'].startswith('tpf:') and int(result['datasource'][4:]) in skip_targets:
					# This secondary target is in the mask of the primary target.
					# We never want to return a lightcurve for a secondary target over
					# a primary target, so we are going to mark this one as SKIPPED.
					primary_tpf_target_starid = int(result['datasource'][4:])
					self.cursor.execute("SELECT priority FROM todolist WHERE starid=? AND datasource='tpf' AND sector=? AND camera=? AND ccd=? AND cadence=?;", (
						primary_tpf_target_starid,
						result['sector'],
						result['camera'],
						result['ccd'],
						result['cadence']
					))
					primary_tpf_target_priority = self.cursor.fetchone()
					# Mark the current star as SKIPPED and that it was caused by the primary:
					self.logger.info("Changing status to SKIPPED for priority %s because it overlaps with primary target TIC %d", result['priority'], primary_tpf_target_starid)
					my_status = STATUS.SKIPPED
					if primary_tpf_target_priority is not None:
						self.cursor.execute("INSERT INTO photometry_skipped (priority,skipped_by) VALUES (?,?);", (
							result['priority'],
							primary_tpf_target_priority[0]
						))
					else:
						self.logger.warning("Could not find primary TPF target (TIC %d) for priority=%d", primary_tpf_target_starid, result['priority'])
						error_msg.append("TargetNotFoundError: Could not find primary TPF target (TIC %d)" % primary_tpf_target_starid)
				else:
					# Create unique list of starids to be masked as skipped:
					skip_starids = ','.join([str(starid) for starid in skip_targets])

					# Ask the todolist if there are any stars that are brighter than this
					# one among the other targets in the mask:
					if result['datasource'] == 'tpf':
						skip_datasources = "'tpf','tpf:%d'" % result['starid']
					else:
						skip_datasources = "'" + result['datasource'] + "'"

					self.cursor.execute("SELECT priority,tmag FROM todolist WHERE starid IN (" + skip_starids + ") AND datasource IN (" + skip_datasources + ") AND sector=? AND camera=? AND ccd=? AND cadence=?;", (
						result['sector'],
						result['camera'],
						result['ccd'],
						result['cadence']
					))
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
								additional_skipped += self.cursor.rowcount
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
							# Mark that the brightest star among the skip-list is the reason for
							# for skipping this target:
							self.cursor.execute("INSERT INTO photometry_skipped (priority,skipped_by) VALUES (?,?);", (
								result['priority'],
								skip_rows[np.argmin(skip_tmags)]['priority']
							))

			# Convert error messages from list to string or None:
			error_msg = None if not error_msg else '\n'.join(error_msg)

			# Update the status in the TODO list:
			self.cursor.execute("UPDATE todolist SET status=? WHERE priority=?;", (
				my_status.value,
				result['priority']
			))

			self.cursor.execute("INSERT OR REPLACE INTO diagnostics (priority, lightcurve, method_used, elaptime, worker_wait_time, pos_column, pos_row, mean_flux, variance, variability, rms_hour, ptp, mask_size, edge_flux, contamination, stamp_width, stamp_height, stamp_resizes, errors) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", (
				result['priority'],
				details.get('filepath_lightcurve', None),
				result['method_used'],
				result['time'],
				result.get('worker_wait_time', None),
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
		except: # noqa: E722, pragma: no cover
			self.conn.rollback()
			raise

		# Update the summary dictionary with the status:
		self.summary['tasks_run'] += 1
		self.summary[my_status.name] += 1
		self.summary['STARTED'] -= 1
		self.summary['SKIPPED'] += additional_skipped

		# Store the last error message in summary:
		if error_msg:
			self.summary['last_error'] = error_msg

		# Calculate mean elapsed time using "streaming weighted mean" with (alpha=0.1):
		# https://dev.to/nestedsoftware/exponential-moving-average-on-streaming-data-4hhl
		if self.summary['mean_elaptime'] is None:
			self.summary['mean_elaptime'] = result['time']
		else:
			self.summary['mean_elaptime'] += 0.1 * (result['time'] - self.summary['mean_elaptime'])

		# All the results should have the same worker_waittime.
		# So only update this once, using just that last result in the list:
		if self.summary['mean_worker_waittime'] is None and result.get('worker_wait_time') is not None:
			self.summary['mean_worker_waittime'] = result['worker_wait_time']
		elif result.get('worker_wait_time') is not None:
			self.summary['mean_worker_waittime'] += 0.1 * (result['worker_wait_time'] - self.summary['mean_worker_waittime'])

		# Write summary file:
		self.summary_counter += 1
		if self.summary_file and self.summary_counter % self.summary_interval == 0:
			self.summary_counter = 0
			self.write_summary()

	#----------------------------------------------------------------------------------------------
	def write_summary(self):
		"""Write summary of progress to file. The summary file will be in JSON format."""
		if hasattr(self, 'summary_file') and self.summary_file:
			try:
				with open(self.summary_file, 'w') as fid:
					json.dump(self.summary, fid)
			except: # noqa: E722, pragma: no cover
				self.logger.exception("Could not write summary file")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A TaskManager which keeps track of which targets to process.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import os.path
import sqlite3
import logging

class TaskManager(object):
	"""
	A TaskManager which keeps track of which targets to process.
	"""

	def __init__(self, todo_file):
		"""
		Initialize the TaskManager which keeps track of which targets to process.

		Parameters:
			todo_file: Path to the TODO-file.

		Raises:
			IOError: If TODO-file could not be found.
		"""

		if os.path.isdir(todo_file):
			todo_file = os.path.join(todo_file, 'todo.sqlite')

		if not os.path.exists(todo_file):
			raise IOError('Could not find TODO-file')

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

		# Reset the status of everything for a new run:
		# TODO: This should obviously be removed once we start running for real
		self.cursor.execute("UPDATE todolist SET status=NULL;")
		self.cursor.execute("DROP TABLE IF EXISTS diagnostics;")
		self.conn.commit()

		# Create table for diagnostics:
		self.cursor.execute("""CREATE TABLE IF NOT EXISTS diagnostics (
			priority INT PRIMARY KEY NOT NULL,
			starid BIGINT NOT NULL,
			elaptime REAL NOT NULL,
			mean_flux DOUBLE PRECISION,
			variance DOUBLE PRECISION,
			mask_size INT,
			pos_row REAL,
			pos_column REAL,
			contamination REAL,
			stamp_resizes INT,
			errors TEXT
		);""")
		self.conn.commit()

		# Reset calculations with status STARTED or ABORT:
		self.cursor.execute("DELETE FROM diagnostics WHERE priority IN (SELECT todolist.priority FROM todolist WHERE status IN (4,6));")
		self.cursor.execute("UPDATE todolist SET status=NULL WHERE status IN (4,6);")
		self.conn.commit()

		# Run a cleanup/optimization of the database before we get started:
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

		self.cursor.execute("SELECT priority,starid,method,camera,ccd,datasource FROM todolist WHERE status IS NULL" + constraints + " ORDER BY priority LIMIT 1;")
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	def get_random_task(self):
		"""
		Get random task to be processed.

		Returns:
			dict or None: Dictionary of settings for task.
		"""
		self.cursor.execute("SELECT priority,starid,method,camera,ccd,datasource FROM todolist WHERE status IS NULL ORDER BY RANDOM() LIMIT 1;")
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	def save_result(self, result):
		"""
		Save results and diagnostics. This will update the TODO list.

		Parameters:
			results (dict): Dictionary of results and diagnostics.
		"""
		# Update the status in the TODO list:
		self.cursor.execute("UPDATE todolist SET status=? WHERE priority=?;", (result['status'].value, result['priority']))

		# Also set status of targets that were marked as "SKIPPED" by this target:
		if 'skip_targets' in result['details'] and len(result['details']['skip_targets']) > 0:
			# Create unique list of starids to be masked as skipped:
			skip_starids = [str(starid) for starid in set(result['details']['skip_targets'])]
			skip_starids = ','.join(skip_starids)
			# Mark them as SKIPPED in the database:
			self.cursor.execute("UPDATE todolist SET status=5 WHERE starid IN (" + skip_starids + ") AND datasource=? AND status IS NULL;", (
				result['datasource'],
			))

		# Save additional diagnostics:
		error_msg = result.get('details', {}).get('errors', None)
		if error_msg: error_msg = '\n'.join(error_msg)
		self.cursor.execute("INSERT INTO diagnostics (priority, starid, elaptime, pos_column, pos_row, mean_flux, variance, mask_size, contamination, stamp_resizes, errors) VALUES (?,?,?,?,?,?,?,?,?,?,?);", (
			result['priority'],
			result['starid'],
			result['time'],
			result['details'].get('pos_centroid', (None, None))[0],
			result['details'].get('pos_centroid', (None, None))[1],
			result['details'].get('mean_flux', None),
			result['details'].get('variance', None),
			result['details'].get('mask_size', None),
			result['details'].get('contamination', None),
			result['details'].get('stamp_resizes', 0),
			error_msg
		))
		self.conn.commit()

	def start_task(self, taskid):
		"""
		Mark a task as STARTED in the TODO-list.
		"""
		self.cursor.execute("UPDATE todolist SET status=6 WHERE priority=?;", (taskid,))
		self.conn.commit()

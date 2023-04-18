#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connection to the central TASOC database.

Note:
	This function requires the user to be connected to the TASOC network
	at Aarhus University. It connects to the TASOC database to get a complete
	list of all stars in the TESS Input Catalog (TIC), which is a very large
	table.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import psycopg
from psycopg.rows import dict_row
import getpass
import contextlib
import random

#--------------------------------------------------------------------------------------------------
class TASOC_DB(object): # pragma: no cover
	"""
	Connection to the central TASOC database.

	Attributes:
		conn (:class:`psycopg.Connection`): Connection to PostgreSQL database.
		cursor (:class:`psycopg.Cursor`): Cursor to use in database.
	"""

	def __init__(self, username=None, password=None):
		"""
		Open connection to central TASOC database.

		If ``username`` or ``password`` is not provided or ``None``,
		the user will be prompted for them.

		Parameters:
			username (string or None, optional): Username for TASOC database.
			password (string or None, optional): Password for TASOC database.
		"""
		if username is None:
			default_username = getpass.getuser()
			username = input('Username [%s]: ' % default_username)
			if username == '':
				username = default_username

		if password is None:
			password = getpass.getpass('Password: ')

		# Open database connection:
		self.conn = psycopg.connect('host=10.28.0.127 user=' + username + ' password=' + password + ' dbname=db_aadc',
			autocommit=False)
		self.cursor = self.conn.cursor(row_factory=dict_row)

	#----------------------------------------------------------------------------------------------
	def close(self):
		self.cursor.close()
		self.conn.close()

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args, **kwargs):
		self.close()

	#----------------------------------------------------------------------------------------------
	def named_cursor(self, name=None, itersize=2000):
		if name is None:
			name = 'tasocdb-{0:06d}'.format(random.randint(0, 999999))
		named_cursor = contextlib.closing(self.conn.cursor(name=name, row_factory=dict_row))
		named_cursor.itersize = itersize
		return named_cursor

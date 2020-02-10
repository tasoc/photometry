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

import psycopg2 as psql
from psycopg2.extras import DictCursor
import getpass

#------------------------------------------------------------------------------
class TASOC_DB(object): # pragma: no cover
	"""
	Connection to the central TASOC database.

	Attributes:
		conn (`psycopg2.Connection` object): Connection to PostgreSQL database.
		cursor (`psycopg2.Cursor` object): Cursor to use in database.
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
		self.conn = psql.connect('host=10.28.0.127 user=' + username + ' password=' + password + ' dbname=db_aadc')
		self.cursor = self.conn.cursor(cursor_factory=DictCursor)

	def close(self):
		self.cursor.close()
		self.conn.close()

	def __enter__(self):
		return self

	def __exit__(self, *args, **kwargs):
		self.close()

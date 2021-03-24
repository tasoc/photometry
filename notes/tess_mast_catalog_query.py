#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs
import astropy.units as u

if __name__ == '__main__':

	buffer_coord = 0.1

	radius = np.sqrt(6**2 + 6**2) + buffer_coord
	radius = u.Quantity(radius, u.deg)
	print(radius)

	catalogData = Catalogs.query_region("352.49324 12.16683", radius=radius, catalog="Tic") # disposition=None

	print(catalogData)


import os.path
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from bottleneck import nanmedian
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.utilities import find_ffi_files
from photometry.plots import plot_image, plots_interactive
from photometry.io import FFIImage

#--------------------------------------------------------------------------------------------------
def detect_bad_smear_columns(fpath, plot=False):

	if isinstance(fpath, str):
		#img = load_ffi_fits(fpath)
		img = FFIImage(fpath)

	ms = nanmedian(img.vsmear, axis=0)
	indx1 = (ms > 250)

	ms2 = nanmedian(img, axis=0)
	indx2 = (ms2 < -1000)

	indx = np.zeros_like(indx1, dtype='bool')
	lab_both, num_both = label(indx1 | indx2)
	for k in range(num_both):
		indx_this = (lab_both == k+1)
		if np.any(indx1 & indx_this):
			indx[indx_this] = True

	if plot:
		img.mask[:, indx] = True

		fig, axes = plt.subplots(5, 1, sharex=True)
		plot_image(img.smear, ax=axes[0], vmin=0, vmax=250, title='smear')
		plot_image(img.vsmear, ax=axes[1], vmin=0, vmax=250, title='vsmear')
		axes[2].stairs(ms, edges=np.arange(-0.5, img.vsmear.shape[1]+0.5))
		axes[2].axhline(250, c='r', ls='--')
		#axes[2].set_ylim(ymax=300)

		axes[3].stairs(ms2, edges=np.arange(-0.5, img.vsmear.shape[1]+0.5))
		axes[3].axhline(-1000, c='r', ls='--')

		plot_image(img, ax=axes[4])

		#axes[0].set_xlim(1640, 1670)
		for ax in axes:
			ax.set_aspect('auto')

		#axes[2].set_ylim(1500, 2048)

	print( np.where(indx)[0] )
	return indx

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plots_interactive()

	indx = detect_bad_smear_columns('./tess2019116075933-s0011-2-2-0143-s_ffic.fits')
	indx = detect_bad_smear_columns('./tess2019116085933-s0011-2-2-0143-s_ffic.fits')

	files = find_ffi_files('../tests/input/images', sector=1)
	for fpath in files:
		print(fpath)
		indx = detect_bad_smear_columns(fpath)
		assert np.all(~indx)

	plt.show()

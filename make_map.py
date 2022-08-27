import healpy as hp
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd

#url = "https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

input_cl = pd.read_csv("COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt",delim_whitespace=True, index_col=0)
lmax = input_cl.index[-1]
cl = input_cl.divide(input_cl.index * (input_cl.index+1) / (np.pi*2), axis="index")
cl = cl.reindex(np.arange(0, lmax+1))
cl = cl.fillna(0)
cl /= 1e12

seed = 420
np.random.seed(seed)
alm = hp.synalm((cl.TT, cl.EE, cl.BB, cl.TE), lmax=lmax, new=True)
high_nside = 2048*2
cmb_map = hp.alm2map(alm, nside=high_nside, lmax=lmax)
hp.fitsfunc.write_map("NSIDE_"+str(high_nside)+".fits",cmb_map)


#hp.mollview(cmb_map[0], min=-300*1e-6, max=300*1e-6, unit="K", title="CMB Temperature")
#plt.ylabel("$\dfrac{\ell(\ell+1)}{2\pi} C_\ell~[\mu K^2]$")
#plt.xlabel("$\ell$")
#plt.xlim([50, 2500]);
#plt.show()
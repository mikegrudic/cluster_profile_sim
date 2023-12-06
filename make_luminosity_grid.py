import numpy as np
from get_isochrone import *
import pickle

filt = "ACS_F555W"
track = "mist_2016_vvcrit_40"

grids = {}
for age in 1e5,1e6,1e7,1e8,1e9,1e10:
    print(age)
    mgrid = np.logspace(-1, np.log10(300), 1000)
    logmgrid = np.log10(mgrid)
    Lgrid = get_luminosity(mgrid, age, filt)
    logLgrid = np.log10(Lgrid)
    grids[age] = mgrid, Lgrid

pickle.dump((filt, track, grids), open(f"mass_lum_grid_{filt}_{track}.dump", "wb"))

imf_samples = np.load("/home/mgrudic/kroupa_m300_samples.npy")
imf_samples = np.float32(imf_samples[: 10**7])
np.save("kroupa_m300_samples.npy", imf_samples)

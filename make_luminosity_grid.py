import numpy as np
from get_isochrone import *
import pickle
from os.path import isdir
from os import mkdir

if not isdir("lumgrids"):
    mkdir("lumgrids")

filters = [
    "WFC3_UVIS_F336W",
    "WFC3_UVIS_F438W",
    "WFC3_UVIS_F555W",
    "WFC3_UVIS_F814W",
    "WFC3_UVIS_F657N",
]
track = "geneva_2013_vvcrit_00"

for f in filters:
    grids = {}
    for age in 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10:
        print(age)
        mgrid = np.logspace(-1, np.log10(300), 1000)
        logmgrid = np.log10(mgrid)
        Lgrid = get_luminosity(mgrid, age, filter=f)
        logLgrid = np.log10(Lgrid)
        grids[age] = mgrid, Lgrid

    pickle.dump((f, track, grids), open(f"mass_lum_grid_{f}_{track}.dump", "wb"))

imf_samples = np.load("/home/mgrudic/kroupa_m300_samples.npy")
imf_samples = np.float32(imf_samples[: 10**7])
np.save("kroupa_m300_samples.npy", imf_samples)

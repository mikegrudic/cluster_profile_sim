import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from get_isochrone import *
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import poisson
from os.path import isfile

Lsol = 3.83e33

tracks = ("geneva_2013_vvcrit_00",
          "geneva_2013_vvcrit_40",
          "geneva_mdot_std",
          "geneva_mdot_enhanced",
          "padova_tpagb_yes",
          "padova_tpagb_no",
          "mist_2016_vvcrit_00",
          "mist_2016_vvcrit_40")
plt.clf()
fig, ax = plt.subplots()
for track in tracks:
    print(track)
    for filter in ("Lbol",):  # "ACS_F555W", "WFC3_UVIS_F555W", "SDSS_g", "SDSS_i", :

        ages = np.logspace(5, 10, 51)
        m50s = []
        mmax = []
        L_over_M = []
        frac = []
        for age in ages:
            m, Lbol, L = generate_random_luminosities(
                10**7, age, return_masses=True, filter=filter, return_Lbol=True, track=track
            )
            print(age, m.mean(), m.sum(), 4.83 - np.log10(L.sum() / Lsol) * 2.5)
            m50s.append(
                10
                ** np.interp(0.5, np.cumsum(L[m.argsort()]) / L.sum(), np.sort(np.log10(m)))
            )
            frac.append(np.sum(m > m50s[-1]) / len(m))
            mmax.append(m.max())
            L_over_M.append(L.sum() / m.sum() / Lsol)
        ax.loglog(ages, L_over_M, label=track)
    ax.legend()
    ax.set(xlabel="Age (yr)", title=filter)
plt.savefig(f"ML_vs_age_{filter}.pdf", bbox_inches="tight")

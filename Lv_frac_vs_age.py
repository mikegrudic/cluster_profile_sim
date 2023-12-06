import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz
from get_isochrone import *
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.stats import poisson
from os.path import isfile

Lsol = 3.83e33

for filter in ("Johnson_V",):  # "ACS_F555W", "WFC3_UVIS_F555W", "SDSS_g", "SDSS_i", :
    plt.clf()
    fig, ax = plt.subplots()
    ages = np.logspace(5, 10, 51)
    m50s = []
    mmax = []
    L_over_M = []
    frac = []
    for age in ages:
        m, Lbol, L = generate_random_luminosities(
            10**6, age, return_masses=True, filter=filter, return_Lbol=True
        )
        print(age, m.mean(), m.sum(), 4.83 - np.log10(L.sum() / Lsol) * 2.5)
        m50s.append(
            10
            ** np.interp(0.5, np.cumsum(L[m.argsort()]) / L.sum(), np.sort(np.log10(m)))
        )
        frac.append(np.sum(m > m50s[-1]) / len(m))
        mmax.append(m.max())
        L_over_M.append(L.sum() / m.sum() / Lsol)
    ax.loglog(ages, mmax, label=r"$M_{\rm max}$")
    ax.loglog(ages, m50s, label=r"$M_{\rm 50}$")
    ax.loglog(ages, frac, label="Fraction of stars emitting 50\% of light")
    ax.loglog(ages, L_over_M, label=r"$L/M$")
    ax.legend()
    ax.set(xlabel="Age (yr)", title=filter)
    plt.savefig(f"Lv_frac_{filter}.pdf", bbox_inches="tight")

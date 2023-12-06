# %%
import pickle
from glob import glob
import numpy as np
from os import system
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sys import argv

bootstrap = int(argv[1])


def fit_results():
    data = np.concatenate([np.load(r) for r in glob("results2/*.npy")])
    (
        gammas,
        backgrounds,
        apertures,
        N,
        res,
        count_photons,
        loga_measured,
        gamma_measured,
    ) = data.T
    # print(len(data))
    count_photons = np.bool_(count_photons)
    r50_0 = (2 ** (2 / (gammas - 2)) - 1) ** 0.5
    r50 = 10**loga_measured * np.sqrt(2 ** (2 / (np.array(gamma_measured) - 2)) - 1)
    dlogr = np.log10(r50 / r50_0)
    cut = (
        np.isfinite(dlogr)
        # * (np.abs(dlogr) < 0.2)
        # * (N > 10**5)
        # *
        # * (~count_photons)
    )
    if bootstrap:
        cut *= np.random.rand(len(cut)) > 0.5

    dlogr = dlogr[cut]
    print(len(dlogr))

    def sigma_dex_model(x):
        Reff_model = r50_0[cut]
        ap_over_Reff = apertures[cut] / Reff_model

        m = (
            10 ** x[0]
            * N[cut] ** -0.5
            #            * (1 + (Reff_SNR / 10 ** x[1]) ** x[2])
            # * (1 + (backgrounds[cut] / 10 ** x[1]) ** x[2])
            # * (1 + (ap_over_Reff / 10 ** x[3]) ** x[4])
            # * (1 + 10 ** x[5] * (gammas[cut] - 2) ** x[6])
        )

        if np.any(count_photons[cut]):
            m[count_photons[cut]] *= 10 ** x[7]
        return m

    def lossfunc(x, *args):
        sigma = sigma_dex_model(x)
        return np.sum(
            (dlogr**2) / (2 * sigma**2) + 0.5 * np.log(2 * np.pi * sigma**2)
        )

    sol = minimize(
        lossfunc,
        [
            -0.83909003,
            -1.51705558,
            0.51937858,
            0.65256137,
            -1.02654235,
            0.3575365,
            -1.92120545,
            0.99351499,
        ],
        method="Nelder-Mead",
        bounds=[
            (-3, 3),
            (np.log10(backgrounds.min()), np.log10(backgrounds.max())),
            (-10, 10),
            (-3, 3),
            (-10, 10),
            (-10, 10),
            (-10, 10),
            (-10, 10),
        ],
        tol=1e-6,
    )
    print(sol.success, sol.fun)
    return sol.x


if bootstrap:
    results = np.array([fit_results() for i in range(bootstrap)])
    # for i in 0, 1, 3, 5, 7:
    # results[:, i] = 10 ** results[:, i]
    sigma = np.diff(np.percentile(results, [16, 84], axis=0), axis=0)[0]
    print(np.c_[np.median(results, axis=0), sigma])
else:
    results = fit_results()
    # for i in 0, 1, 3, 5, 7:
    # results[i] = 10 ** results[i]
    print(results)

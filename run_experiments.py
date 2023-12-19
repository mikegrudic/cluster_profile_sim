from scipy.optimize import minimize, newton
from os.path import isdir
from numba import vectorize
from time import time
import os
import pickle
from joblib import Parallel, delayed
from get_isochrone import *
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from lossfuncs import *

model = argv[1]

# os.environ["OMP_NUM_THREADS"] = "1"
# global parameters that will apply to every model
filt = "ACS_F555W"
track = "mist_2016_vvcrit_40"
age_yr = 1e8


def mass_to_lum(mass, logmgrid, logLgrid):
    return 10 ** np.interp(np.log10(mass), logmgrid, logLgrid)


filt, track, grids = pickle.load(
    open(f"lumgrids/mass_lum_grid_{filt}_{track}.dump", "rb")
)
mgrid, Lgrid = grids[age_yr]
logmgrid = np.log10(mgrid)
logLgrid = np.log10(Lgrid)
mmax = mgrid[np.isfinite(logLgrid)].max()

imf_samples = np.load("kroupa_m300_samples.npy")
imf_samples = imf_samples[imf_samples < mmax]

cluster_avg_lighttomass = (
    mass_to_lum(imf_samples, logmgrid, logLgrid).sum() / imf_samples.sum()
)
field_avg_lighttomass = 0.5
m_avg_cluster = imf_samples.mean()


def inference_experiment(
    shape=2.5,
    background=1e-6,
    aperture=30,
    N=10**4,
    res=0.1,
    count_photons=False,
    dist_mpc=1,
    method="poisson",
    exposure_s=1000,
    model="EFF",
    full_output=False,
    num_bins=300,
):
    N = round(N)
    cluster_radii, background_radii = generate_radii(
        shape, background, aperture, N, model, count_photons
    )
    if np.any(np.isnan(cluster_radii)):
        return [np.nan, np.nan]
    radii = np.concatenate([cluster_radii, background_radii])
    if count_photons:
        res = max(res, 0.5 * 1.94e-7 * dist_mpc * 1e6)
    rbins = np.logspace(
        max(np.log10(res), -1), np.log10(aperture), min(N // 2, num_bins)
    )
    rbins[0] = 0
    N = len(cluster_radii)
    if count_photons:  # mock hubble photon counts
        masses_cluster = np.random.choice(imf_samples, len(cluster_radii))
        L = mass_to_lum(masses_cluster, logmgrid, logLgrid)
        # masses_bg = np.random.choice(imf_samples_field, len(background_radii))
        # L_bg = mass_to_lum(masses_bg, logmgrid_field, logLgrid_field)
        # L = np.concatenate([L_cluster, L_bg])
        Q = L / 3.579e-12  # photons per second
        # photons expected from each star: Q * t * effective area / (4 pi r^2)
        photons_expected = 3.78e-48 * (dist_mpc / 10) ** -2 * Q * exposure_s
        if method == "djorgovski87":
            radii_split = np.array_split(radii, 8)  # split into 8 sectors
            photons_expected_split = np.array_split(photons_expected, 8)
            photons_perbin_expected = np.array(
                [
                    np.histogram(r, rbins, weights=p)[0]
                    for r, p in zip(radii_split, photons_expected_split)
                ]
            )

            photons_perbin_expected += (
                background
                * central_norm(shape, model)
                * N
                * cluster_avg_lighttomass
                * m_avg_cluster
                * np.diff(np.pi * rbins**2)
                / 3.579e-12
                * 3.78e-48
                * (dist_mpc / 10) ** -2
                * exposure_s
                / 8
            )
            bin_counts = [
                np.random.poisson(P, size=P.shape) for P in photons_perbin_expected
            ]

            mu0_est = max(
                photons_perbin_expected.sum(0)[rbins[1:] < 0.5].sum()
                / (np.pi * 0.5**2),
                1,
            )
        else:
            photons_perbin_expected = np.histogram(
                cluster_radii, rbins, weights=photons_expected
            )[0]
            # add smooth background, emulating an older stellar pop
            # photons_perbin_expected += (
            #     background
            #     * central_norm(shape, model)
            #     * N
            #     * cluster_avg_lighttomass
            #     * m_avg_cluster
            #     * np.diff(np.pi * rbins**2)
            #     / 3.579e-12
            #     * 3.78e-48
            #     * (dist_mpc / 10) ** -2
            #     * exposure_s
            # )

            bin_counts = np.random.poisson(
                photons_perbin_expected, size=photons_perbin_expected.shape
            )
            mu0_est = max(
                photons_perbin_expected[rbins[1:] < 0.5].sum() / (np.pi * 0.5**2), 1
            )
    else:
        if method == "djorgovski87":
            radii_split = np.array_split(radii, 8)
            bin_counts = [np.histogram(r, rbins)[0] for r in radii_split]
        else:
            bin_counts = np.histogram(radii, rbins)[0]
        mu0_est = min(10, N) / (np.pi * radii[min(9, N - 1)] ** 2)

    if method == "poisson":
        lossfunc_touse = lossfunc
    else:
        lossfunc_touse = lossfunc_djorgovski87
    # plt.loglog(
    #     np.sqrt(rbins[1:] * rbins[:-1]),
    #     bin_counts / np.diff(np.pi * rbins**2),
    # )
    p0 = np.array(
        [
            max(-10, np.log10(mu0_est)),
            max(-10, np.log10(background * mu0_est)),
            0.0,
            np.log10(shape),
        ]
    )

    if model == "EFF":
        shape_range = np.log10([1e-2, 20])
    elif model == "King62":
        shape_range = np.log10([1e-3, 10 * aperture])
    bounds = [
        (np.log10(N) - 6, np.log10(N) + 6),
        (-10, 10),
        (-10, 1 + np.log10(aperture)),
        shape_range,
    ]

    fac = 1e-2
    sol_best = p0
    fun_best = lossfunc_touse(p0, rbins, bin_counts, model)
    for i in range(100):
        guess = np.array(sol_best) + fac * np.random.normal(size=(4,))

        sol = minimize(
            lossfunc_touse,
            guess,
            args=(rbins, bin_counts, model),
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "xatol": 1e-6,
            },
        )
        fac *= 1.05
        if sol.fun < fun_best:
            sol_best = sol.x
            fun_best = sol.fun
        if sol.success:
            break

    sol.x[-1] = 10 ** sol.x[-1]

    if sol.success:
        return sol.x if full_output else sol.x[2:]
    else:
        return 4 * [np.nan] if full_output else 2 * [np.nan]


@vectorize
def king62_cdf(x, c):
    if x >= c:
        return 1.0
    fac = 1 / (1 + c * c)
    norm = -3 - fac + 4 * fac**0.5 + np.log(1 + c * c)
    cdf = (
        x**2 * fac - 4 * (np.sqrt(1 + x**2) - 1) * fac**0.5 + np.log(1 + x**2)
    ) / norm
    return cdf


def king62_r50(rc, c, tol=1e-13):
    # good initial guess:
    try:
        r0 = 0.5 * min(c**0.5, c)
        return rc * newton(lambda x: king62_cdf(x, c) - 0.5, r0, tol=tol)
    except:
        rgrid = np.logspace(-3, 0, 10000)
        return np.interp(0.5, king62_cdf(rgrid, c), rgrid) * rc


def king62_central_norm(c):
    """Returns the central surface density of a normalized King 1962 profile of concentration c with unit core radius"""
    fac = (1 + c * c) ** -0.5
    norm = (fac - 1) ** 2 / (np.pi * (np.log(1 + c**2) - 3 - fac * fac + 4 * fac))
    return norm


def eff_central_norm(gamma):
    """Returns the central surface density of a normalized EFF profile of with slope gamma and unit scale radius"""
    return (gamma - 2) / (2 * np.pi)


def central_norm(shape, model="EFF"):
    if model == "EFF":
        return eff_central_norm(shape)
    else:
        return king62_central_norm(shape)


def generate_radii(
    shape=2.5, background=1e-1, Rmax=100, N=100, model="EFF", count_photons=False
):
    np.random.seed()
    if model == "EFF":
        gamma = shape
        norm = (gamma - 2) / (2 * np.pi)
        r_cluster = np.sort(np.sqrt((1 - np.random.rand(N)) ** (-2 / (gamma - 2)) - 1))
        r50_target = (2 ** (2 / (gamma - 2)) - 1) ** 0.5
    elif model == "King62":
        c = shape
        x = np.linspace(0, c, 100000)
        cdf = king62_cdf(x, c)
        r_cluster = np.sort(np.interp(np.random.rand(N), cdf, x))
        r50_target = king62_r50(1.0, c)  # np.interp(0.5, cdf, x)
        norm = king62_central_norm(c)

    r_cluster *= r50_target / 10 ** np.interp(
        0.5, np.linspace(0, 1, N), np.log10(r_cluster)
    )
    r_cluster = r_cluster[r_cluster < Rmax]

    sigma_background = norm * background
    if count_photons:
        N_background = 0
    else:
        N_background = round(sigma_background * np.pi * Rmax**2 * N)
    r_background = Rmax * np.sqrt(np.random.rand(N_background))
    return r_cluster, r_background


def generate_parameter_grid(num_params=10**2, model="EFF"):
    """Generates a list of parameters for the simulated cluster+background inference experiments"""

    # uniform between 10^3-10^6 stars in the cluster
    Ncluster = np.int_(10 ** (2 + 4 * np.random.rand(num_params)))
    # uniform  distribution of gammas between 2.1 and 4.1
    if model == "EFF":
        gammas = 2.1 + 2 * np.random.rand(num_params)
    elif model == "King62":
        gammas = 10 ** (2 * np.random.rand(num_params))
    # loguniform distribution of aperture sizes between 3 and 300 scale radii
    apertures = 10 ** (np.log10(3) + np.log10(100 / 3) * np.random.rand(num_params))
    # backgrounds: bound between 1e-5 and 1e-1
    backgrounds = 10 ** (3 * np.random.rand(num_params)) / Ncluster
    # resolution: fix at 0.1 scale radii
    res = np.repeat(0.1, num_params)
    # do a coin flip to decide whether we're counting photons or counting
    count_photons = np.random.rand(num_params) > 0.5
    # ages: fix to 1e8 (should be similar after 1e7yr
    # ages = np.repeat(1e8, num_params)

    return zip(gammas, backgrounds, apertures, Ncluster, res, count_photons)


def main():
    #    model = "King62
    if not isdir(f"results_{model}"):
        os.mkdir(f"./results_{model}")
    while True:
        params = generate_parameter_grid(100, model)
        Nsamples = 10**3
        ts = []
        for p in params:
            t = time()
            result = np.array(
                Parallel(10)(
                    delayed(inference_experiment)(*p, model=model)
                    for i in range(Nsamples)
                )
            )
            print(p, np.median(result, axis=0))
            pickle.dump((p, result), open(f"results_{model}/{hash(p)}.dump", "wb"))
            ts.append(time() - t)


#        break

if __name__ == "__main__":
    main()

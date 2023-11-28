import numpy as np
from scipy.optimize import minimize
from os.path import isdir
from numba import vectorize
from time import time
import os
import pickle
from joblib import Parallel, delayed
from get_isochrone import *


# os.environ["OMP_NUM_THREADS"] = "1"

# global parameters that will apply to every model
filt = "ACS_F555W"
age_yr = 1e8

filt, track, grids = open(f"mass_lum_grid_{filt}_{track}.dump", "rb")
mgrid, Lgrid = grids[age_yr]

# mgrid = np.logspace(-1, np.log10(150), 1000)
# logmgrid = np.log10(mgrid)
# Lgrid = get_luminosity(mgrid, 1e8, filt)
# logLgrid = np.log10(Lgrid)
# mmax = mgrid[np.isfinite(logLgrid)].max()
imf_samples = np.load("kroupa_m300_samples.npy")
# imf_samples = imf_samples[imf_samples < mmax][: 10**7]


def mass_to_lum(mass):
    return 10 ** np.interp(np.log10(mass), logmgrid, logLgrid)


def inference_experiment(
    gamma=2.5,
    background=1e-1,
    aperture=30,
    N=10**4,
    res=0.1,
    count_photons=False,
    dist_mpc=10,
    method="poisson",
    exposure_s=1000,
):
    N = round(N)
    cluster_radii, background_radii = generate_radii(
        gamma, background, aperture, N, 1.0
    )
    if np.any(np.isnan(cluster_radii)):
        return [np.nan, np.nan]
    radii = np.concatenate([cluster_radii, background_radii])
    # print(N, len(radii))
    if count_photons:
        res = max(res, 0.5 * 1.94e-7 * dist_mpc * 1e6)
    rbins = np.linspace(0, aperture, int(aperture / res))
    N = len(radii)
    if count_photons:  # mock hubble photon counts
        masses = np.random.choice(imf_samples, N)
        # generate_random_luminosities(N, age, filt=filt) # F555W luminosity
        L = mass_to_lum(masses)
        Q = L / 3.579e-12  # photons per second
        # photons expected from each star: Q * t * effective area / (4 pi r^2)
        photons_expected = 3.78e-48 * (dist_mpc / 10) ** -2 * Q * exposure_s
        if method == "djorgovski87":
            radii_split = np.array_split(radii, 8)  # split into 8 sectors
            photons_expected_split = np.array_split(photons_expected, 8)
            photons_perbin_expected = [
                np.histogram(r, rbins, weights=p)[0]
                for r, p in zip(radii_split, photons_expected_split)
            ]
            N = [np.random.poisson(P, size=P.shape) for P in photons_perbin_expected]
        else:
            photons_perbin_expected = np.histogram(
                radii, rbins, weights=photons_expected
            )[0]
            N = np.random.poisson(
                photons_perbin_expected, size=photons_perbin_expected.shape
            )
        mu0_est = photons_expected.sum() / 2 / (np.pi * np.median(radii) ** 2)
    else:
        if method == "djorgovski87":
            radii_split = np.array_split(radii, 8)
            N = [np.histogram(r, rbins)[0] for r in radii_split]
        else:
            N = np.histogram(radii, rbins)[0]
        mu0_est = 3 / (np.pi * radii[2] ** 2)

    if method == "poisson":
        lossfunc_touse = lossfunc
    else:
        lossfunc_touse = lossfunc_djorgovski87

    p0 = (np.log10(mu0_est), np.log10(background * mu0_est), 0.0, gamma)
    fac = 1e-2
    for i in range(100):
        sol = minimize(
            lossfunc_touse,
            np.array(p0)
            + fac * np.random.rand(4),  # * (0 if i > 0 else np.array([1, 1, 0, 1])),
            args=(rbins, N),
            method="Nelder-Mead",
            bounds=[(-6, 6), (-10, 6), (-10, 1 + np.log10(aperture)), (0, 20)],
            tol=1e-3,
        )
        fac *= 1.05
        if sol.success:
            break
    if sol.success:
        return sol.x[2:]
    else:
        return [np.nan, np.nan]


@vectorize(fastmath=True)
def logpoisson(counts, expected_counts):
    """Fast computation of log of poisson PMF, using Ramanujan's Stirling-type approximation to factorial"""
    if counts == 0:
        logfact = 0
    elif counts < 10:
        fact = counts
        for i in range(2, counts):  # factorial
            fact *= i
        logfact = np.log(fact)
    else:  # stirling-type approximation due to Ramanujan
        logfact = (
            np.log(counts) * counts
            - counts
            + np.log(counts * (1 + 4 * counts * (1 + 2 * counts))) / 6
            + np.log(np.pi) * 0.5
        )
    return counts * np.log(expected_counts) - expected_counts - logfact


def generate_radii(gamma=2.5, background=1e-1, Rmax=100, N=100, a=1):
    np.random.seed()
    norm = (gamma - 2) / (2 * np.pi)
    sigma_background = norm * background
    r_cluster = np.sort(np.sqrt((1 - np.random.rand(N)) ** (-2 / (gamma - 2)) - 1))
    r50_desired = (2 ** (2 / (gamma - 2)) - 1) ** 0.5
    r_cluster *= r50_desired / np.interp(0.5, np.linspace(0, 1, N), r_cluster)
    r_cluster = np.sort(r_cluster[r_cluster < Rmax])
    N_background = round(sigma_background * np.pi * Rmax**2 * N)
    r_background = Rmax * np.sqrt(np.random.rand(N_background))
    return r_cluster, r_background


def lossfunc(x, rbins, bincounts):
    logmu0, logbackground, loga, gam = x
    mu, bg, a = 10**logmu0, 10**logbackground, 10**loga
    cumcounts_avg = (
        2
        * a**gam
        * np.pi
        * (a ** (2 - gam) - (a**2 + rbins**2) ** (1 - gam / 2))
        * mu
        / (gam - 2)
        + np.pi * rbins**2 * bg
    )
    expected_counts = np.diff(cumcounts_avg)
    prob = logpoisson(bincounts, expected_counts).sum()
    return -prob


def lossfunc_djorgovski87(x, rbins, bincounts):
    """
    Surface brightness profile fitting loss function using the method of Djorgovski 1987
    for estimating error-bars in radial surface brightness/number density bins:
    divide into 8 sectors and compute standard deviation within each annulus (effectively bootstrapping)
    """
    logmu0, logbackground, loga, gam = x
    mu, bg, a = 10**logmu0, 10**logbackground, 10**loga
    cumcounts_avg = (
        2
        * a**gam
        * np.pi
        * (a ** (2 - gam) - (a**2 + rbins**2) ** (1 - 0.5 * gam))
        * mu
        / (gam - 2)
        + np.pi * rbins**2 * bg
    )
    expected_counts = np.diff(cumcounts_avg)
    stderr = np.std(bincounts, axis=0)
    N = np.sum(bincounts, axis=0)
    prob = np.sum(-((expected_counts - N) ** 2) / (2 * stderr**2))
    return -prob


def generate_parameter_grid(num_params=10**2):
    """Generates a list of parameters for the simulated cluster+background inference experiments"""

    # uniform between 10^3-10^6 stars in the cluster
    Ncluster = np.int_(10 ** (3 + 3 * np.random.rand(num_params)) + 0.5)
    # uniform  distribution of gammas between 2.1 and 4.1
    gammas = 2.1 + 2 * np.random.rand(num_params)
    # loguniform distribution of aperture sizes between 3 and 300 scale radii
    apertures = 10 ** (np.log10(3) + 2 * np.random.rand(num_params))
    # backgrounds: bound between 1/N and 1000/N
    backgrounds = 10 ** (3 * np.random.rand(num_params)) / Ncluster
    # resolution: fix at 0.1 scale radii
    res = np.repeat(0.1, num_params)
    # do a coin flip to decide whether we're counting photons or counting
    count_photons = np.random.rand(num_params) > 0.5
    # ages: fix to 1e8 (should be similar after 1e7yr
    # ages = np.repeat(1e8, num_params)

    return zip(gammas, backgrounds, apertures, Ncluster, res, count_photons)


def main():
    if not isdir("results"):
        os.mkdir("./results")
    while True:
        params = generate_parameter_grid(100)
        Nsamples = 10**4
        ts = []
        for p in params:
            t = time()
            print(p)
            result = np.array(
                Parallel(35)(delayed(inference_experiment)(*p) for i in range(Nsamples))
            )
            pickle.dump((p, result), open(f"results/{hash(p)}.dump", "wb"))
            ts.append(time() - t)


#        break

if __name__ == "__main__":
    main()

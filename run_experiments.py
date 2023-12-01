import numpy as np
from scipy.optimize import minimize
from os.path import isdir
from numba import vectorize
from time import time
import os
import pickle
from joblib import Parallel, delayed
from get_isochrone import *
from matplotlib import pyplot as plt


# os.environ["OMP_NUM_THREADS"] = "1"

# global parameters that will apply to every model
filt = "ACS_F555W"
track = "mist_2016_vvcrit_40"
age_yr = 1e8

filt, track, grids = pickle.load(open(f"mass_lum_grid_{filt}_{track}.dump", "rb"))
mgrid, Lgrid = grids[age_yr]
logmgrid = np.log10(mgrid)
logLgrid = np.log10(Lgrid)
mmax = mgrid[np.isfinite(logLgrid)].max()
imf_samples = np.load("kroupa_m300_samples.npy")
imf_samples = imf_samples[imf_samples < mmax]


def mass_to_lum(mass):
    return 10 ** np.interp(np.log10(mass), logmgrid, logLgrid)


def inference_experiment(
    shape=2.5,
    background=1e-6,
    aperture=30,
    N=10**4,
    res=0.1,
    count_photons=False,
    dist_mpc=10,
    method="poisson",
    exposure_s=1000,
    model="EFF",
):
    N = round(N)
    cluster_radii, background_radii = generate_radii(
        shape, background, aperture, N, model
    )
    if np.any(np.isnan(cluster_radii)):
        return [np.nan, np.nan]
    radii = np.concatenate([cluster_radii, background_radii])
    if count_photons:
        res = max(res, 0.5 * 1.94e-7 * dist_mpc * 1e6)
    rbins = np.logspace(max(np.log10(res), -1), max(2, np.log10(aperture)), 1000)
    rbins[0] = 0
    # rbins = np.linspace(0, aperture, int(aperture / res))
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

    p0 = (np.log10(mu0_est), np.log10(background * mu0_est), 0.0, shape)
    fac = 1e-2
    if model == "EFF":
        shape_range = (0, 20)
    elif model == "King62":
        shape_range = (1e-3, 10 * aperture)

    for i in range(100):
        sol = minimize(
            lossfunc_touse,
            np.array(p0)
            + fac * np.random.rand(4),  # * (0 if i > 0 else np.array([1, 1, 0, 1])),
            args=(rbins, N, model),
            method="Nelder-Mead",
            bounds=[(-6, 6), (-10, 6), (-10, 1 + np.log10(aperture)), shape_range],
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


def king62_cdf(x, c):
    fac = 1 / (1 + c * c)
    norm = -3 - fac + 4 * fac**0.5 + np.log(1 + c * c)
    cdf = (
        x**2 * fac - 4 * (np.sqrt(1 + x**2) - 1) * fac**0.5 + np.log(1 + x**2)
    ) / norm
    cdf[x > c] = 1
    return cdf


def king62_r50(rc, c):
    rgrid = np.logspace(-3, np.log10(c), 10000) * rc
    return np.interp(0.5, king62_cdf(rgrid / rc, c), rgrid)


def king62_central_norm(c):
    fac = (1 + c * c) ** -0.5
    norm = (fac - 1) ** 2 / (np.pi * (np.log(1 + c**2) - 3 - fac * fac + 4 * fac))
    return norm


def generate_radii(shape=2.5, background=1e-1, Rmax=100, N=100, model="EFF"):
    np.random.seed()
    if model == "EFF":
        gamma = shape
        norm = (gamma - 2) / (2 * np.pi)
        r_cluster = np.sort(np.sqrt((1 - np.random.rand(N)) ** (-2 / (gamma - 2)) - 1))
        r50_target = (2 ** (2 / (gamma - 2)) - 1) ** 0.5
    elif model == "King62":
        c = shape
        x = np.linspace(0, c, 10000)
        cdf = king62_cdf(x, c)
        norm = king62_central_norm(c)
        r_cluster = np.sort(np.interp(np.random.rand(N), cdf, x))
        r50_target = king62_r50(1.0, c)  # np.interp(0.5, cdf, x)

    r_cluster *= r50_target / np.interp(0.5, np.linspace(0, 1, N), r_cluster)
    r_cluster = r_cluster[r_cluster < Rmax]

    sigma_background = norm * background
    N_background = round(sigma_background * np.pi * Rmax**2 * N)
    r_background = Rmax * np.sqrt(np.random.rand(N_background))
    return r_cluster, r_background


def lossfunc(x, rbins, bincounts, model="EFF"):
    logmu0, logbackground, loga, shape = x
    mu, bg, a = 10**logmu0, 10**logbackground, 10**loga
    if model == "EFF":
        gam = shape
        cumcounts_avg = (
            mu
            * 2
            * a**gam
            * np.pi
            * (a ** (2 - gam) - (a**2 + rbins**2) ** (1 - gam / 2))
            / (gam - 2)
            + np.pi * rbins**2 * bg
        )
    elif model == "King62":
        c = shape
        cumcounts_avg = mu * king62_cdf(rbins, c) + np.pi * rbins**2 * bg
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


def generate_parameter_grid(num_params=10**2, model="EFF"):
    """Generates a list of parameters for the simulated cluster+background inference experiments"""

    # uniform between 10^3-10^6 stars in the cluster
    Ncluster = np.int_(10 ** (3 + 3 * np.random.rand(num_params)) + 0.5)
    # uniform  distribution of gammas between 2.1 and 4.1
    if model == "EFF":
        gammas = 2.1 + 2 * np.random.rand(num_params)
    elif model == "King62":
        gammas = 10 ** (2 * np.random.rand(num_params))
    # loguniform distribution of aperture sizes between 3 and 300 scale radii
    apertures = 10 ** (np.log10(3) + np.log10(100 / 3) * np.random.rand(num_params))
    # backgrounds: bound between 1/N and 1000/N
    backgrounds = 10 ** (3.5 * np.random.rand(num_params)) / Ncluster
    # resolution: fix at 0.1 scale radii
    res = np.repeat(0.1, num_params)
    # do a coin flip to decide whether we're counting photons or counting
    count_photons = np.random.rand(num_params) > 0.5
    # ages: fix to 1e8 (should be similar after 1e7yr
    # ages = np.repeat(1e8, num_params)

    return zip(gammas, backgrounds, apertures, Ncluster, res, count_photons)


def main():
    model = "King62"
    if not isdir(f"results_{model}"):
        os.mkdir(f"./results_{model}")
    while True:
        params = generate_parameter_grid(100, model)
        Nsamples = 10**3
        ts = []
        for p in params:
            t = time()
            print(p)
            result = np.array(
                Parallel(18)(
                    delayed(inference_experiment)(*p, model=model)
                    for i in range(Nsamples)
                )
            )
            print(np.median(result, axis=0))
            pickle.dump((p, result), open(f"results_{model}/{hash(p)}.dump", "wb"))
            ts.append(time() - t)


#        break

if __name__ == "__main__":
    main()

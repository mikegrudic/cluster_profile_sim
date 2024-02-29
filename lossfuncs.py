"""Loss functions for density profile fitting"""

import numpy as np
from density_models import *
from numba import vectorize
from scipy.stats import median_abs_deviation


@vectorize(fastmath=True)
def logpoisson(counts, expected_counts):
    """Fast computation of log of poisson PMF, using Ramanujan's Stirling-type
    approximation to factorial"""
    counts = int(counts + 0.5)
    if expected_counts == 0:
        if counts == 0:
            return 0
        else:
            return -np.inf
    if counts == 0:
        logfact = 0
    elif counts < 10:
        fact = counts
        for i in range(2, counts):  # factorial
            fact *= i
        logfact = np.log(fact)
    else:  # stirling-type approximation due to Ramanujan
        # cast counts to avoid integer overflow inside 3rd order term
        counts = float(counts)
        logfact = (
            np.log(counts) * counts
            - counts
            + np.log(counts * (1 + 4 * counts * (1 + 2 * counts))) / 6
            + np.log(np.pi) * 0.5
        )
    return counts * np.log(expected_counts) - expected_counts - logfact


def lossfunc(x, rbins, bincounts, model="EFF"):
    if np.any(np.isnan(x)):
        return np.inf
    if len(x) == 4:
        logmu0, logbackground, loga, logshape = x
        mu, bg, a, shape = 10**logmu0, 10**logbackground, 10**loga, 10**logshape
    else:
        # logmu0, logbackground, loga, logshape, logcutoff = x
        mu, bg, a, shape, cutoff = 10**x

    if model == "EFF":
        gam = 10**logshape
        cumcounts_avg = (
            mu
            * 2
            * a**gam
            * np.pi
            * (a ** (2 - gam) - (a**2 + rbins**2) ** (1 - gam / 2))
            / (gam - 2)
            + np.pi * rbins**2 * bg
        )
    elif model == "EFF_cutoff":
        cumcounts_avg = (
            mu * EFF_cutoff_cdf(rbins / a, shape, cutoff / a) + np.pi * rbins**2 * bg
        )
    elif model == "King62":
        c = 10**logshape
        cumcounts_avg = mu * king62_cdf(rbins / a, c) + np.pi * rbins**2 * bg
    expected_counts = np.diff(cumcounts_avg)

    prob = logpoisson(bincounts, expected_counts).sum()
    return -prob


def lossfunc_djorgovski87(x, rbins, bincounts, model="EFF"):
    """
    Surface brightness profile fitting loss function using the method of Djorgovski 1987
    for estimating error-bars in radial surface brightness/number density bins:
    divide into 8 sectors and compute standard deviation within each annulus (effectively bootstrapping)
    """
    if np.any(np.isnan(x)):
        return np.inf
    logmu0, logbackground, loga, logshape = x
    mu, bg, a, shape = 10**logmu0, 10**logbackground, 10**loga, 10**logshape
    # cumcounts_avg = mu * model_cdf(rbins / a, shape, model) + np.pi * rbins**2 * bg
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
        cumcounts_avg = mu * king62_cdf(rbins / a, c) + np.pi * rbins**2 * bg
    area = np.diff(np.pi * rbins**2)
    expected_counts = np.diff(cumcounts_avg)
    # total_counts = np.median(bincounts, axis=0)  # * 8 / area
    # stderr = median_abs_deviation(bincounts, axis=0) * 1.4826  # * np.sqrt(8)
    total_counts = np.sum(bincounts, axis=0)
    stderr = np.std(bincounts, axis=0)
    # print(np.c_[total_counts, total_counts2])
    prob = -np.sum(
        (((expected_counts / 8 - total_counts) ** 2) / (2 * stderr**2))[(stderr > 0)]
    )
    # print(prob)
    return -prob

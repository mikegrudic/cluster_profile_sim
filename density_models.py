"""Helper functions for working with various star cluster density profiles"""

from scipy.optimize import newton
from numba import vectorize
import numpy as np


def king62_central_norm(c):
    """Returns the central surface density of a normalized King 1962 profile of
    concentration c with unit core radius"""
    fac = (1 + c * c) ** -0.5
    norm = (fac - 1) ** 2 / (np.pi * (np.log(1 + c**2) - 3 - fac * fac + 4 * fac))
    return norm


def eff_central_norm(gamma):
    """Returns the central surface density of a normalized EFF profile of with
    slope gamma and unit scale radius"""
    return (gamma - 2) / (2 * np.pi)


def central_norm(shape, scale_radius=1.0, model="EFF"):
    """Returns the central value of a projected number density model
    *normalized to 1*"""
    if model == "EFF":
        return eff_central_norm(shape) / scale_radius**2
    else:
        return king62_central_norm(shape) / scale_radius**2


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


def king62_r50(c, scale_radius=1.0, tol=1e-13):
    # good initial guess:
    try:
        r0 = 0.5 * min(c**0.5, c)
        return scale_radius * newton(lambda x: king62_cdf(x, c) - 0.5, r0, tol=tol)
    except:
        rgrid = np.logspace(-3, 0, 10000)
        return np.interp(0.5, king62_cdf(rgrid, c), rgrid) * scale_radius


def EFF_cdf(x, gamma):
    return np.sqrt((1 - x) ** (-2 / (gamma - 2)) - 1)


def EFF_r50(gamma, scale_radius=1.0):
    return np.sqrt(2 ** (2 / (gamma - 2)) - 1) * scale_radius


def r50(shape, scale_radius=1.0, model="EFF"):
    if model == "EFF":
        return EFF_r50(shape, scale_radius)
    elif model == "King62":
        return king62_r50(shape, scale_radius)

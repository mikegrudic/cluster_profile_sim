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


def eff_cutoff_central_norm(gamma, x_cut=np.inf):
    """Returns the central surface density of a normalized EFF profile of with
    slope gamma unit scale radius and a cutoff equal to x_cut * scale radius"""
    return (gamma - 2) / (2 * np.pi) / ((1 + x_cut**2) ** (1 - 0.5 * gamma) - 1)


def central_norm(shape, scale_radius=1.0, cutoff_radius=None, model="EFF"):
    """Returns the central value of a projected number density model
    *normalized to 1*"""
    if model == "EFF":
        return eff_central_norm(shape) / scale_radius**2
    elif model == "EFF_cutoff":
        return (
            eff_cutoff_central_norm(shape, cutoff_radius / scale_radius)
            / scale_radius**2
        )
    elif model == "King62":
        return king62_central_norm(shape) / scale_radius**2
    else:
        raise NotImplementedError(f"Density model {model} has not been implemented.")


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


def EFF_inv_cdf(x, gamma):
    """Inverse of the CDF of an EFF profile with unit scale radius."""
    return np.sqrt((1 - x) ** (-2 / (gamma - 2)) - 1)


def EFF_cdf(x, gamma):
    """CDF of an EFF profile with unit scale radius"""
    return 1 - (1 + x * x) ** (1 - 0.5 * gamma)


def EFF_r50(gamma, scale_radius=1.0):
    """Reff of an EFF profile"""
    return np.sqrt(2 ** (2 / (gamma - 2)) - 1) * scale_radius


def EFF_cutoff_r50(gamma, scale_radius=1.0, cutoff_radius=np.inf):
    """Reff of a truncated EFF profile"""
    xmax = cutoff_radius / scale_radius
    return scale_radius * (
        2 ** (1 / (-2 + gamma))
        * (1 + (1 + xmax**2) ** (1 - gamma / 2.0)) ** (1 / (2 - gamma))
        * np.sqrt(
            1
            - 4 ** (1 / (2 - gamma))
            * (1 + (1 + xmax**2) ** (1 - gamma / 2.0)) ** (2 / (-2 + gamma))
        )
    )


@vectorize
def EFF_cutoff_cdf(x, gamma, x_cutoff=np.inf):
    """CDF of a truncated EFF profile with unit scale radius"""
    # xmax = cutoff_radius / scale_radius
    if x > x_cutoff:
        return 1.0
    fx = (1 + x**2) ** (1 - 0.5 * gamma)
    fxmax = (1 + x_cutoff**2) ** (1 - 0.5 * gamma)
    frac = (fx - 1) / (fxmax - 1)
    return frac


def model_r50(shape, scale_radius=1.0, model="EFF", cutoff_radius=np.inf):
    if model == "EFF":
        return EFF_r50(shape, scale_radius)
    elif model == "EFF_cutoff":
        return EFF_cutoff_r50(shape, scale_radius, cutoff_radius)
    elif model == "King62":
        return king62_r50(shape, scale_radius)
    else:
        raise NotImplementedError(f"Density model {model} has not been implemented.")


def mass_aperture_fac(shape, scale_radius, aperture, model="EFF", cutoff_radius=np.inf):
    if model == "EFF":
        if shape <= 2:
            return np.inf
        else:
            return EFF_cdf(np.inf, shape) / EFF_cdf(aperture / scale_radius, shape)
    elif model == "EFF_cutoff":
        return EFF_cutoff_cdf(np.inf, shape, cutoff_radius) / EFF_cutoff_cdf(
            aperture / scale_radius, shape, aperture
        )
    elif model == "King62":
        return king62_cdf(np.inf, shape) / king62_cdf(aperture / scale_radius, shape)
    else:
        raise NotImplementedError(f"Density model {model} has not been implemented.")

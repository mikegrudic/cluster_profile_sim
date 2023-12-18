import numpy as np
from os import system
from os.path import isfile
import subprocess
from io import StringIO
from numba import njit, prange
import pickle

wfc3_filters = (
    "WFC3_UVIS_F218W",
    "WFC3_UVIS_F225W",
    "WFC3_UVIS_F275W",
    "WFC3_UVIS_F336W",
    "WFC3_UVIS_F360W",
    "WFC3_UVIS_F438W",
    "WFC3_UVIS_F475W",
    "WFC3_UVIS_F555W",
    "WFC3_UVIS_F606W",
    "WFC3_UVIS_F775W",
    "WFC3_UVIS_F814W",
)


def get_isochrone(
    time_yr,
    filters=("ACS_F555W",),
    track="mist_2016_vvcrit_40",
):
    """
    Returns a data table of photometric quantities as a function of stellar mass, at a given age
    Column 0: initial mass
    Column 1: current mass
    Column 2: logL
    Column 3: logT
    Column 4: logR
    Column 5: log g
    Column 6: WR
    Column 7+: Luminosities in specified filters, in erg/s
    """

    path = (
        f"/tmp/isochrone_logage{np.log10(time_yr)}_{track}" + " ".join(filters) + ".dat"
    )
    if not isfile(path):
        system(
            f"$SLUG_DIR/tools/c/write_isochrone/write_isochrone {track} {time_yr} -f "
            + " ".join(filters)
            + " > "
            + path
        )
    return np.genfromtxt(path, skip_header=4, skip_footer=2)


def generate_isochrone_grid(
    ages,
    filters=wfc3_filters,
    track="geneva_2013_vvcrit_00",
    mmin=0.1,
    mmax=120,
    num_masses=100,
):
    """From a grid of ages in yr, generates a grid of stellar properties including
    photometry in the specified filters using the specified evolution track

    Shape: (# of ages, # of masses, # of bands + 7)
    """
    result = str(
        subprocess.check_output(
            f"$SLUG_DIR/tools/c/write_isochrone/write_isochrone {track} "
            + " ".join(str(a) for a in ages)
            + " ".join((" -f " + f for f in filters))
            + f" -nm {num_masses} -m0 {mmin} -m1 {mmax}",
            shell=True,
        )
    ).replace("\\n", "\n")

    grids = np.array(
        [np.loadtxt(StringIO(a.split("---")[0])) for a in result.split("--\n")[1::2]]
    )

    with open("isochrone_grid", "wb") as f:
        pickle.dump((filters, ages, grids), f)
    return ages, grids


@njit(fastmath=True, error_model="numpy")
def log_interp_indices_and_weights(xgrid, x):
    logx = np.log10(x)
    logx_max, logx_min = np.log10(xgrid[-1]), np.log10(xgrid[0])
    if (logx < logx_min) or (logx > logx_max):
        return -1, -1, np.nan, np.nan
    dlogx = (logx_max - logx_min) / (xgrid.shape[0] - 1)

    idx1 = max(int((logx - logx_min) / dlogx), 0)
    idx2 = min(idx1 + 1, xgrid.shape[0] - 1)
    wt2 = (logx - np.log10(xgrid[idx1])) / dlogx
    wt1 = 1 - wt2
    return idx1, idx2, wt1, wt2


@njit(fastmath=True, error_model="numpy")
def log_interpolant(xgrid, x, y):
    idx1, idx2, wt1, wt2 = log_interp_indices_and_weights(xgrid, x)
    if np.isnan(wt1) or np.isnan(wt2):
        return np.nan * y[idx1]
    return wt1 * np.log10(y[idx1]) + wt2 * np.log10(y[idx2])


@njit(fastmath=True, error_model="numpy")
def lin_interpolant(xgrid, x, y):
    idx1, idx2, wt1, wt2 = log_interp_indices_and_weights(xgrid, x)
    if np.isnan(wt1) or np.isnan(wt2):
        return np.nan * y[idx1]
    return wt1 * y[idx1] + wt2 * y[idx2]


@njit(parallel=True)
def get_photometry_of_stars(masses, ages, agegrid, isochrone_grid, magnitudes=True):
    num_filters = isochrone_grid.shape[2] - 7

    if magnitudes:
        interpolator = lin_interpolant
        result = 100 * np.ones((masses.shape[0], num_filters), dtype=np.float64)
    else:
        interpolator = log_interpolant
        result = np.zeros((masses.shape[0], num_filters), dtype=np.float64)

    for i in prange(masses.shape[0]):
        mass, age = masses[i], ages[i]
        age_idx1, age_idx2, wt_age1, wt_age2 = log_interp_indices_and_weights(
            agegrid, age
        )
        if np.isnan(wt_age1) or np.isnan(wt_age2):
            continue
        mgrid1 = isochrone_grid[age_idx1, :, 0]
        mgrid2 = isochrone_grid[age_idx2, :, 0]
        interpolant1 = interpolator(mgrid1, mass, isochrone_grid[age_idx1, :, 7:])
        interpolant2 = interpolator(mgrid2, mass, isochrone_grid[age_idx2, :, 7:])

        if magnitudes:
            val = wt_age1 * interpolant1 + wt_age2 * interpolant2
        else:
            val = 10 ** (wt_age1 * interpolant1 + wt_age2 * interpolant2)

        for j in range(num_filters):
            if np.isfinite(val[j]):
                result[i, j] = val[j]

    return result


def get_luminosity(mass, age, filter, track="geneva_2013_vvcrit_00"):
    iso = get_isochrone(age, filters=[filter], track=track)
    mgrid = iso[:, 0]
    Lgrid = iso[:, 7]

    L = 10 ** np.interp(np.log10(mass), np.log10(mgrid), np.log10(Lgrid))
    L[mass > mgrid.max()] = 0
    return L


def generate_random_luminosities(
    N,
    age,
    filter="ACS_F555W",
    return_masses=False,
    return_Lbol=False,
    track="geneva_2013_vvcrit_00",
):
    iso = get_isochrone(
        age,
        filters=[
            (filter if filter else "ACS_F555W"),
        ],
        track=track,
    )
    mgrid = iso[:, 0]
    Lbolgrid = 10 ** iso[:, 2] * 3.83e33
    if filter:
        Lgrid = iso[:, 7]
    else:
        Lgrid = Lbol
    m = np.random.choice(imf_samples, 2 * N)
    m = m[m < mgrid.max()][:N]
    L = 10 ** np.interp(np.log10(m), np.log10(mgrid), np.log10(Lgrid))

    if return_Lbol:
        Lbol = 10 ** np.interp(np.log10(m), np.log10(mgrid), np.log10(Lbolgrid))
        if return_masses:
            return m, Lbol, L
        else:
            return Lbol, L

    if return_masses:
        return m, L
    else:
        return L

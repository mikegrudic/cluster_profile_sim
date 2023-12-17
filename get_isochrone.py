import numpy as np
from os import system
from os.path import isfile
import subprocess
from io import StringIO
from numba import vectorize

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
    return grids


# @vectorize
# def get_photometry_of_stars(masses,ages,isochrone_grid):


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

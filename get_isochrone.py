import numpy as np
from os import system
from os.path import isfile

mmax = 150
mmin = 1
imf_samples = np.load("/home/mgrudic/kroupa_m300_samples.npy")[: 10**8]
imf_samples = imf_samples[(imf_samples < mmax)]


def get_isochrone(
    time_yr,
    filters=[
        "ACS_F555W",
    ],
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


def get_luminosity(mass, age, filter, track="mist_2016_vvcrit_40"):
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
    track="mist_2016_vvcrit_40",
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

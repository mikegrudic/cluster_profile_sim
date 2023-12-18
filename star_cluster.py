from os.path import isfile
import pickle
import numpy as np
from density_models import *
from get_isochrone import generate_isochrone_grid, get_photometry_of_stars
from filterlists import messa_m51_filters, chandar_m83_filters
from slugpy.cluster_slug import cluster_slug


MMAX = 100
IMF_SAMPLES = np.load("kroupa_m300_samples.npy")
IMF_SAMPLES = IMF_SAMPLES[IMF_SAMPLES < MMAX]


# def mass_to_lum(mass, logmgrid, logLgrid):
# return 10 ** np.interp(np.log10(mass), logmgrid, logLgrid)


class StarCluster:
    """Randomly-generated star cluster sampled from a prescribed number density
    distribution, optionally with masses sampled from the IMF"""

    def __init__(
        self,
        num_stars,
        scale_radius=1.0,
        model="EFF",
        shape=2.5,
        rmax=100,
        background=0,
        seed=None,
        # sample_masses=False,
        # filters=None,
    ):
        """Initializes the basic properties of the star cluster and calls
        generate_radii to randomly sample the cluster radii"""
        self.num_stars = num_stars
        self.scale_radius = scale_radius
        self.r50_target = r50(shape, scale_radius)
        self.norm = central_norm(shape, scale_radius, model)
        self.density_model = model
        self.shape = shape
        self.rmax = rmax
        self.seed = seed
        self.background = background
        self.cluster_radii = self.background_radii = self.num_background = None
        self.masses = self.initial_mass = None
        self.photometry = {}

    def get_cluster_radii(self, enforce_r50=False):
        """Randomly samples projected stellar radii from the specified number density model"""
        if self.cluster_radii is not None:
            return self.cluster_radii

        np.random.seed(self.seed)

        if self.density_model == "EFF":
            self.cluster_radii = np.sort(
                EFF_cdf(np.random.rand(self.num_stars), self.shape)
            )
        elif self.density_model == "King62":
            c = self.shape
            x = np.linspace(0, c, 100000)
            cdf = king62_cdf(x, c)
            self.cluster_radii = np.sort(
                np.interp(np.random.rand(self.num_stars), cdf, x)
            )

        if enforce_r50:
            r50_target = r50(self.shape, self.scale_radius, self.density_model)
            self.cluster_radii *= r50_target / 10 ** np.interp(
                0.5, np.linspace(0, 1, self.num_stars), np.log10(self.cluster_radii)
            )
        self.cluster_radii = self.cluster_radii[self.cluster_radii < self.rmax]
        return self.cluster_radii

    def get_background_radii(self):
        """Samples positions of background stars"""
        if self.background_radii is not None:
            return self.background_radii

        if isinstance(self.seed, int):
            np.random.seed(self.seed + 1)
        sigma_background = self.norm * self.background
        num_background = round(
            sigma_background * np.pi * self.rmax**2 * self.num_stars
        )
        self.background_radii = self.rmax * np.sqrt(np.random.rand(num_background))
        return self.background_radii

    def get_all_radii(self):
        """Returns the radii of both the cluster and background stars"""
        return self.get_cluster_radii(), self.get_background_radii()

    def initial_stellar_masses(self):
        """Samples masses of stars in the cluster"""
        if self.masses is None:
            self.masses = np.random.choice(IMF_SAMPLES, self.num_stars)
        return self.masses

    def get_photometry(
        self,
        ages,
        filters=messa_m51_filters,
        track="geneva_2013_vvcrit_00",
        return_sum=False,
    ):
        """Compute photometry values for individual stars"""

        # get stellar masses
        masses = self.initial_stellar_masses()
        if isinstance(ages, float):
            ages = np.repeat(ages, len(masses))

        agegrid = np.logspace(5, 10, 1001)

        # look for photometry grid and check it has the filters we need
        if not isfile("isochrone_grid"):
            grids = generate_isochrone_grid(agegrid, filters, track=track)
        else:
            with open("isochrone_grid", "rb") as f:
                grid_track, grid_filters, grid_ages, grids = pickle.load(f)
            if (
                (set(filters) != set(grid_filters))
                or (np.any(grid_ages != agegrid))
                or (track != grid_track)
            ):
                print("Generating isochrone grid for desired filters...")
                grids = generate_isochrone_grid(agegrid, filters, track=track)
                print("Done!")

        phot = get_photometry_of_stars(
            masses,
            ages,
            agegrid,
            grids,
            magnitudes=True,
        )
        # self.live_stars =
        if return_sum:  # add up the magnitudes
            return -2.5 * np.log10(np.sum(10 ** (-phot / 2.5), axis=0))
        return phot

    def measure_slug(
        self,
        ages,
        measurement="Mass",
        cs=None,
        filters=messa_m51_filters,
        track="geneva_2013_vvcrit_00",
    ):
        if cs is None:
            cs = cluster_slug(use_nebular=False, photsystem="Vega", filters=filters)

        phot = self.get_photometry(ages, filters, track, return_sum=True)

        logx, pdf = cs.mpdf({"age": 1, "mass": 0}[measurement.lower()], phot)
        logx_lower, logx_med, logx_upper = np.interp(
            [0.16, 0.5, 0.84], pdf.cumsum() / pdf.sum(), logx
        )
        return logx_lower, logx_med, logx_upper

    def measure_radius(self, count_photons=True):
        if count_photons:
            phot = self.get_photometry(ages, filters, track)

    def binned_density_profile(self, num_bins=300, res=0.1):
        """Returns the effective bin radii and values of the binned projected
        number density profile"""
        rbins = (
            np.logspace(
                max(np.log10(res), -1),
                np.log10(self.rmax),
                min(self.num_stars // 3, num_bins),
            )
            * self.scale_radius
        )
        counts = np.histogram(np.concatenate(self.get_all_radii()), rbins)[0]
        r_eff = np.sqrt(rbins[1:] * rbins[:-1])
        number_density = counts / np.diff(np.pi * rbins**2)
        return r_eff, number_density

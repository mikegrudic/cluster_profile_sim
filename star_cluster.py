"""Defines StarCluster class"""

from os.path import isfile
import pickle
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from density_models import *
from get_isochrone import generate_isochrone_grid, get_photometry_of_stars
from filterlists import messa_m51_filters
from slugpy.cluster_slug import cluster_slug
from lossfuncs import *


MMAX = 100
IMF_SAMPLES = np.load("kroupa_samples.npy")
IMF_SAMPLES = IMF_SAMPLES[IMF_SAMPLES < MMAX]


class StarCluster:
    """Randomly-generated star cluster sampled from a prescribed number density
    distribution, optionally with masses sampled from the IMF"""

    def __init__(
        self,
        num_stars,
        scale_radius=1.0,
        model="EFF",
        shape=2.5,
        background=0,
        seed=None,
        cutoff=np.inf,
        # sample_masses=False,
        # filters=None,
    ):
        """Initializes the basic properties of the star cluster and calls
        generate_radii to randomly sample the cluster radii"""
        self.num_stars = num_stars
        self.scale_radius = scale_radius
        self.cutoff = cutoff  # the ACTUAL ground-truth cutoff, not the aperture radius!
        self.r50_target = model_r50(shape, scale_radius)
        self.norm = central_norm(shape, scale_radius, model)
        self.density_model = model
        self.shape = shape
        self.seed = seed
        self.background = background
        self.cluster_radii = self.background_radii = self.num_background = []
        self.masses = self.initial_mass = None
        self.photometry = {}

    def get_cluster_radii(self, enforce_r50=False):
        """Randomly samples projected stellar radii from the specified number density model"""
        if self.cluster_radii != []:
            return self.cluster_radii

        np.random.seed(self.seed)

        x_rand = np.random.rand(self.num_stars)
        if self.density_model == "EFF":
            self.cluster_radii = (
                np.sort(EFF_inv_cdf(x_rand, self.shape)) * self.scale_radius
            )
        elif self.density_model == "EFF_cutoff":
            x = np.linspace(0, self.cutoff, 100000)
            cdf = EFF_cutoff_cdf(x, self.shape, self.cutoff)
            self.cluster_radii = np.sort(np.interp(x_rand, cdf, x))
        elif self.density_model == "King62":
            c = self.shape
            x = np.linspace(0, c, 100000)
            cdf = king62_cdf(x, c)
            self.cluster_radii = np.sort(np.interp(x_rand, cdf, x))

        if enforce_r50:
            r50_target = model_r50(self.shape, self.scale_radius, self.density_model)
            self.cluster_radii *= r50_target / 10 ** np.interp(
                0.5, np.linspace(0, 1, self.num_stars), np.log10(self.cluster_radii)
            )
        self.cluster_radii = self.cluster_radii  # [self.cluster_radii < self.rmax]
        return self.cluster_radii

    def get_background_radii(self, rmax=15):
        """Samples positions of background stars"""
        if self.background_radii != []:
            return self.background_radii

        if isinstance(self.seed, int):
            np.random.seed(self.seed + 1)
        sigma_background = self.norm * self.background
        num_background = round(sigma_background * np.pi * rmax**2 * self.num_stars)
        self.background_radii = rmax * np.sqrt(np.random.rand(num_background))
        return self.background_radii

    def get_all_radii(self, rmax=15):
        """Returns the radii of both the cluster and background stars"""
        return self.get_cluster_radii(), self.get_background_radii(rmax)

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
        aperture=None,
    ):
        """Compute photometry values for individual stars"""

        # get stellar masses
        masses = self.initial_stellar_masses()
        if aperture:
            cut = self.cluster_radii < aperture
            masses = masses[cut]
        if isinstance(ages, float):
            ages = np.repeat(ages, len(masses))

        agegrid = np.logspace(5, 10, 1001)

        # look for photometry grid and check it has the filters we need
        if not isfile("isochrone_grid"):
            print("Generating isochrone grid for desired filters...")
            grids = generate_isochrone_grid(agegrid, filters, track=track)
            print("Done!")
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

    def measure_bayesian(
        self,
        ages,
        measurement="Mass",
        cs=None,
        filters=messa_m51_filters,
        track="geneva_2013_vvcrit_00",
        aperture=None,
    ):
        if cs is None:
            cs = cluster_slug(use_nebular=False, photsystem="Vega", filters=filters)

        phot = self.get_photometry(
            ages, filters, track, return_sum=True, aperture=aperture
        )

        logx, pdf = cs.mpdf({"age": 1, "mass": 0, "av": 2}[measurement.lower()], phot)
        logx_lower, logx_med, logx_upper = np.interp(
            [0.16, 0.5, 0.84], pdf.cumsum() / pdf.sum(), logx
        )
        return logx_lower, logx_med, logx_upper

    def fit_density_profile(
        self,
        count_photons=False,
        aperture=15,
        num_bins=100,
        method="poisson",
        res=1e-1,
        age=3e8,
        dist_mpc=1,
        model=None,
    ):
        if model is None:
            model = self.density_model
        N = self.num_stars
        N_eff = N // 100 if count_photons else N
        cluster_radii = self.get_cluster_radii()
        if np.any(np.isnan(cluster_radii)):
            return [np.nan, np.nan]

        if count_photons:
            all_radii = np.copy(cluster_radii)
        else:
            all_radii = np.concatenate(self.get_all_radii(rmax=aperture))
        rcut = all_radii < aperture
        all_radii = all_radii[rcut]
        # if count_photons:
        #     res = max(res, 0.5 * 1.94e-7 * dist_mpc * 1e6)
        rbins = np.logspace(
            max(np.log10(res), -1), np.log10(aperture), min(N_eff // 2, num_bins)
        )
        rbins[0] = 0
        if count_photons:  # mock hubble photon counts
            # just treat ACS F555W as Johnson V
            phot = self.get_photometry(age)[:, 3][rcut]
            lum_solar = 10 ** ((4.8 - phot) / 2.5)
            # light_to_mass = lum_solar /
            lum_cgs = 4e33 * lum_solar
            Q = lum_cgs / 3.579e-12  # photons per second for 555nm

            # photons expected from each star: Q * t * effective area / (4 pi r^2)
            exposure_s = 1000
            photons_expected = 3.78e-48 * (dist_mpc / 10) ** -2 * Q * exposure_s

            if method == "djorgovski87":
                radii_split = np.array_split(all_radii, 8)  # split into 8 sectors
                photons_expected_split = np.array_split(photons_expected, 8)
                photons_perbin_expected = np.array(
                    [
                        np.histogram(r, rbins, weights=p)[0]
                        for r, p in zip(radii_split, photons_expected_split)
                    ]
                )

                photons_perbin_expected += (
                    self.background
                    * self.norm
                    * lum_cgs.sum()
                    * np.diff(np.pi * rbins**2)
                    / 3.579e-12
                    * 3.78e-48
                    * (dist_mpc / 10) ** -2
                    * exposure_s
                ) / 8
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
                    all_radii, rbins, weights=photons_expected
                )[0]
                # add smooth background, emulating an older stellar pop
                photons_perbin_expected += (
                    self.background
                    * self.norm
                    * lum_cgs.sum()
                    * np.diff(np.pi * rbins**2)
                    / 3.579e-12
                    * 3.78e-48
                    * (dist_mpc / 10) ** -2
                    * exposure_s
                )

                bin_counts = np.random.poisson(
                    photons_perbin_expected, size=photons_perbin_expected.shape
                )
                # plt.loglog(rbins[1:], bin_counts / np.diff(rbins**2))
                # plt.show()
                mu0_est = max(
                    photons_perbin_expected[rbins[1:] < 0.5].sum() / (np.pi * 0.5**2),
                    1,
                )
        else:
            if method == "djorgovski87":
                radii_split = np.array_split(all_radii, 8)
                bin_counts = [np.histogram(r, rbins)[0] for r in radii_split]
            else:
                bin_counts = np.histogram(all_radii, rbins)[0]
            mu0_est = min(10, N) / (np.pi * all_radii[min(9, N - 1)] ** 2)

        if method == "poisson":
            lossfunc_touse = lossfunc
        else:
            lossfunc_touse = lossfunc_djorgovski87

        p0 = [
            max(-10, np.log10(mu0_est)),
            max(-10, np.log10(self.background * mu0_est)),
            np.log10(self.scale_radius),
            np.log10(self.shape),
        ]
        if "cutoff" in model:
            p0.append(np.log10(self.cutoff))

        p0 = np.array(p0)
        n_params = len(p0)

        if "EFF" == model:
            shape_range = np.log10([2, 1e6])
        elif model == "EFF_cutoff":
            shape_range = np.log10([0.1, 1e6])
        elif model == "King62":
            shape_range = np.log10([1e-3, 10 * aperture])
        bounds = [
            (np.log10(N) - 6, np.log10(N) + 6),
            (-10, 10),
            (-10, 1 + np.log10(aperture)),
            shape_range,
        ]
        if "cutoff" in model:
            bounds.append([np.log10(self.cutoff) - 2, np.log10(aperture)])

        fac = 1e-2
        sol_best = p0
        fun_best = lossfunc_touse(p0, rbins, bin_counts, model)
        for _ in range(100):
            guess = np.array(sol_best) + fac * np.random.normal(size=(n_params,))

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

        # shape parameter
        sol.x = 10**sol.x

        if sol.success:
            return sol.x  # if full_output else sol.x[2:]
        return n_params * [np.nan]  # if full_output else 2 * [np.nan]

    def get_concentration_index(self, res=0.1):
        return 0

    def measure_r50(
        self, count_photons=False, age=3e8, method="poisson", model=None, aperture=15
    ):
        if model is None:
            model = self.density_model
        params = self.fit_density_profile(
            count_photons=count_photons,
            age=age,
            method=method,
            model=model,
            aperture=aperture,
        )
        # print(params)
        scale_radius = params[2]
        shape = params[3]
        if "cutoff" in model:
            cutoff = params[-1]
        else:
            cutoff = np.inf
        return model_r50(shape, scale_radius, self.density_model, cutoff_radius=cutoff)

    def binned_density_profile(
        self, num_bins=300, res=0.1, aperture=15, count_photons=False
    ):
        """Returns the effective bin radii and values of the binned projected
        number density profile"""
        rbins = (
            np.logspace(
                max(np.log10(res), -1),
                np.log10(aperture),
                min(self.num_stars // 3, num_bins),
            )
            * self.scale_radius
        )
        # if count_photons:

        counts = np.histogram(np.concatenate(self.get_all_radii(rmax=aperture)), rbins)[
            0
        ]
        r_eff = np.sqrt(rbins[1:] * rbins[:-1])
        number_density = counts / np.diff(np.pi * rbins**2)
        return r_eff, number_density

    def plot_density_profile(self, num_bins=300, res=0.1, aperture=15):
        r, sigma = self.binned_density_profile(num_bins, res, aperture=aperture)
        # params = self.fit_density_profile()

        plt.loglog(r, sigma)
        plt.show()

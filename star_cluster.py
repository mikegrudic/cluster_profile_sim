"""Defines StarCluster class"""

from os.path import isfile
import pickle
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from density_models import *
from get_isochrone import generate_isochrone_grid, get_photometry_of_stars
from filterlists import messa_m51_filters

# from slugpy.cluster_slug import cluster_slug
from lossfuncs import *


MMAX = 100
IMF_SAMPLES = np.load("kroupa_samples.npy")
IMF_SAMPLES = IMF_SAMPLES[IMF_SAMPLES < MMAX]
MEAN_STELLAR_MASS = IMF_SAMPLES.mean()


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

    def get_cluster_radii(self, enforce_r50=True):
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
        elif self.density_model == "Gaussian":
            x = np.random.normal(size=(self.num_stars, 2))
            self.cluster_radii = np.sort((x * x).sum(1) ** 0.5)

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
        mass_background_msun = np.pi * rmax**2 * self.background
        num_background = round(mass_background_msun / MEAN_STELLAR_MASS)
        # sigma_background = self.norm * self.background
        # num_background = round(sigma_background * np.pi * rmax**2 * self.num_stars)
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

        # logx, pdf = cs.mpdf((0, 1), phot)
        # logm, logage = logx
        # logm = logm[:, 0]
        # logage = logage[0, :]
        # m_pdf = pdf.sum(1)
        # age_pdf = pdf.sum(0)

        # logage_lower, logage_med, logage_upper = np.interp(
        #     [0.16, 0.5, 0.84], age_pdf.cumsum() / age_pdf.sum(), logage
        # )
        # logm_lower, logm_med, logm_upper = np.interp(
        #     [0.16, 0.5, 0.84], m_pdf.cumsum() / m_pdf.sum(), logm
        # )
        # # return logx_lower, logx_med, logx_upper
        # if measurement.lower() == "mass":
        #     return [logm_lower, logm_med, logm_upper]
        # if measurement.lower() == "age":
        #     return [logage_lower, logage_med, logage_upper]
        # return {
        #     "logm": [logm_lower, logm_med, logm_upper],
        #     "logage": [logage_lower, logage_med, logage_upper],
        # }

    def dist_to_res_pc(self, dist_mpc):
        return 0.193 * dist_mpc  # WFC3 UVIS pixels

    def fit_density_profile(
        self,
        count_photons=False,
        aperture=15,
        method="poisson",
        res=1e-1,
        age=3e8,
        dist_mpc=1,
        model=None,
        max_num_bins=30,
        num_bins=None,
    ):
        if model is None:
            model = self.density_model
        N = self.num_stars
        cluster_radii = self.get_cluster_radii()
        if np.any(np.isnan(cluster_radii)):
            return [np.nan, np.nan, np.nan, np.nan]

        if count_photons:
            all_radii = np.copy(cluster_radii)
        else:
            all_radii = np.concatenate(self.get_all_radii(rmax=aperture))
        rcut = all_radii < aperture
        all_radii = all_radii[rcut]
        # if #count_photons:
        res = max(
            res, self.dist_to_res_pc(dist_mpc)
        )  # max(res, 0.5 * 1.94e-7 * dist_mpc * 1e6)
        if num_bins:
            rbins = np.linspace(0, aperture, num_bins)
        else:
            rbins = np.linspace(
                0, aperture, min(int(aperture / res) + 1, max_num_bins - 1)
            )  # np.logspace(
        rbins[0] = 0
        # print(rbins)
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

            mass_to_light_old = 1.0  # msun lsun^-1
            surface_brightness_old = (
                self.background / mass_to_light_old * 3.83e33
            )  # erg s^-1 pc^-2

            if method == "djorgovski87":
                # p = np.random.permutation(len(all_radii))
                sector = np.random.randint(0, 8, len(all_radii))
                radii_split = [all_radii[sector == i] for i in range(8)]
                photons_expected_split = [
                    photons_expected[sector == i] for i in range(8)
                ]
                photons_perbin_expected = np.array(
                    [
                        np.histogram(r, rbins, weights=p)[0]
                        for r, p in zip(radii_split, photons_expected_split)
                    ]
                )
                # print(photons_perbin_expected)

                photons_perbin_expected += (
                    surface_brightness_old
                    * np.diff(np.pi * rbins**2)
                    / 3.579e-12
                    * 3.78e-48
                    * (dist_mpc / 10) ** -2  # distance
                    * exposure_s  # exposure time
                ) / 8
                bin_counts = [
                    np.random.poisson(P, size=P.shape) for P in photons_perbin_expected
                ]

                mu0_est = max(
                    photons_perbin_expected.sum(0)[0] / (np.pi * rbins[1] ** 2),
                    1,
                )
            else:
                photons_perbin_expected = np.histogram(
                    all_radii, rbins, weights=photons_expected
                )[0]
                # add smooth background, emulating an older stellar pop
                photons_perbin_expected += (
                    surface_brightness_old
                    * np.diff(np.pi * rbins**2)
                    / 3.579e-12
                    * 3.78e-48
                    * (dist_mpc / 10) ** -2  # distance
                    * exposure_s  # exposure time
                )
                bin_counts = np.random.poisson(
                    photons_perbin_expected, size=photons_perbin_expected.shape
                )
                mu0_est = max(
                    photons_perbin_expected[0] / (np.pi * rbins[1] ** 2),
                    1,
                )
        else:
            if method == "djorgovski87":
                sector = np.random.randint(0, 8, len(all_radii))
                radii_split = [all_radii[sector == i] for i in range(8)]
                bin_counts = [np.histogram(r, rbins)[0] for r in radii_split]
            else:
                bin_counts = np.histogram(all_radii, rbins)[0]
            mu0_est = min(10, N) / (np.pi * all_radii[min(9, N - 1)] ** 2)

        if method == "poisson":
            lossfunc_touse = lossfunc
        else:
            lossfunc_touse = lossfunc_djorgovski87

        p0 = [
            max(-10, np.log10(N)),
            max(-10, np.log10(self.background * mu0_est)),
            np.log10(self.scale_radius),
        ]
        if model != "Gaussian":
            p0.append(np.log10(self.shape))
        if "cutoff" in model:
            p0.append(np.log10(self.cutoff))

        p0 = np.array(p0)
        n_params = len(p0)

        if "EFF" == model:
            shape_range = np.log10([0.1, 1e6])
        elif model == "EFF_cutoff":
            shape_range = np.log10([0.1, 1e6])
        elif model == "King62":
            shape_range = np.log10([1e-3, 10 * aperture])
        bounds = [
            (np.log10(N) - 10, np.log10(N) + 10),
            (-10, 10),
            (-10, 1 + np.log10(aperture)),
        ]
        if model != "Gaussian":
            bounds.append(shape_range)
        if "cutoff" in model:
            bounds.append([np.log10(self.cutoff) - 2, np.log10(aperture)])

        fac = 1e-2
        sol_best = np.copy(p0)
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
                    "xatol": 1e-9,
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

    def concentration_index(self, age=1e8, dist_mpc=5, num_pixels=(1, 3)):
        if dist_mpc < 1:
            return 100.0
        phot = self.get_photometry(age)[:, 3]
        lum_solar = 10 ** ((4.8 - phot) / 2.5)
        radii = self.get_cluster_radii()
        res = self.dist_to_res_pc(dist_mpc)
        pixels = radii / res
        ratio = (
            lum_solar[pixels < num_pixels[0]].sum()
            / lum_solar[pixels < num_pixels[1]].sum()
        )
        return -np.log10(ratio) * 2.5

    def measure_r50(
        self,
        count_photons=False,
        age=3e8,
        method="poisson",
        model=None,
        aperture=15,
        dist_mpc=1,
        max_num_bins=30,
        res=0.1,
        num_bins=None,
    ):
        if model is None:
            model = self.density_model
        params = self.fit_density_profile(
            count_photons=count_photons,
            age=age,
            method=method,
            model=model,
            aperture=aperture,
            dist_mpc=dist_mpc,
            max_num_bins=max_num_bins,
            res=res,
            num_bins=num_bins,
        )
        #        print(params)
        scale_radius = params[2]
        if len(params) > 3:
            shape = params[3]
        else:
            shape = np.nan
        if "cutoff" in model:
            cutoff = params[-1]
        else:
            cutoff = np.inf
        return model_r50(shape, scale_radius, model, cutoff_radius=cutoff)

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

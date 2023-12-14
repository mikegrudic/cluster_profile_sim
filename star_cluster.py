import numpy as np
from density_models import *


class StarCluster:
    """Randomly-generated star cluster sampled from a prescribed number density distribution, optionally with masses sampled from the IMF"""

    def __init__(
        self,
        num_stars,
        age,
        scale_radius,
        model="EFF",
        shape=2.5,
        rmax=100,
        background=0,
        seed=None,
    ):
        """Initializes the basic properties of the star cluster and calls generate_radii to randomly sample the cluster radii"""
        # self.mass = mass
        self.num_stars = num_stars
        self.age = age
        self.scale_radius = scale_radius
        self.r50_target = r50(shape, scale_radius)
        self.norm = central_norm(shape, scale_radius)
        self.model = model
        self.shape = shape
        self.rmax = rmax
        self.seed = seed
        self.background = background
        np.random.seed(seed)
        self.cluster_radii = self.background_radii = None

    def generate_cluster_radii(self):
        """Randomly samples projected stellar radii from the specified number density model"""
        if self.model == "EFF":
            # gamma = self.shape
            # norm = (gamma - 2) / (2 * np.pi)
            self.cluster_radii = np.sort(
                EFF_cdf(np.random.rand(self.num_stars), self.shape)
            )
        elif self.model == "King62":
            c = self.shape
            x = np.linspace(0, c, 100000)
            cdf = king62_cdf(x, c)
            self.cluster_radii = np.sort(
                np.interp(np.random.rand(self.num_stars), cdf, x)
            )
        #    norm = king62_central_norm(c)

        # cluster_radii *= r50_target / 10 ** np.interp(
        #     0.5, np.linspace(0, 1, N), np.log10(cluster_radii)
        # )
        self.cluster_radii = self.cluster_radii[self.cluster_radii < self.rmax]
        return self.cluster_radii

    def generate_background_radii(self):
        sigma_background = self.norm * self.background
        N_background = round(sigma_background * np.pi * self.rmax**2 * self.num_stars)
        self.background_radii = self.rmax * np.sqrt(np.random.rand(N_background))
        return self.background_radii

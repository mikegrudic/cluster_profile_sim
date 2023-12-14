"""Experiments with effect of cluster radius uncertainties on mass-radius relation"""
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from os.path import isfile
from palettable.colorbrewer.qualitative import Dark2_3
from cycler import cycler
from run_experiments import *

ML_age, ML = np.loadtxt("Bruzual_Charlot_2003_ML_vs_Age.csv").T


# for i in range(100):
def generate_masses_and_radii(
    N_clusters=100,
    Mmin=10**3,
    Mmax=10**6,
    slope=0.3333333333,
    cluster_age=1e8,
    cluster_mass_error_dex=0.3,
    shape=2.5,
    model="EFF",
    sigma_star=10,
    Reff_error=True,
):
    # N_clusters =
    Mmin, Mmax = 1e3, 10**7
    x = np.random.rand(N_clusters)
    cluster_masses = Mmax * Mmin / (Mmax * (1 - x) + Mmin * x)
    # add errors to cluster masses
    cluster_masses_measured = cluster_masses * 10 ** (
        cluster_mass_error_dex * np.random.normal(size=cluster_masses.shape)
    )
    cut = cluster_masses_measured > 1e4
    cluster_masses = cluster_masses[cut]
    cluster_masses_measured = cluster_masses_measured[cut]

    Reff_true = (cluster_masses / 1e3) ** slope
    if model == "EFF":
        a_true = Reff_true / (2 ** (2 / (shape - 2)) - 1) ** 0.5
    else:
        a_true = Reff_true / king62_r50(1.0, shape)
    sigma_center = central_norm(shape, model) * cluster_masses / a_true**2
    imf_avg_mass = 0.5
    N_stars = np.int_(cluster_masses / imf_avg_mass)

    # sigma_star_background_msun_pc2 = 10.0
    mass_to_light_field = np.interp(
        10, np.log10(ML_age), ML
    )  # 10Gyr population for field
    mass_to_light_cluster = np.interp(np.log10(cluster_age), np.log10(ML_age), ML)
    backgrounds = (
        sigma_star / sigma_center * mass_to_light_cluster / mass_to_light_field
    )
    # print(mass_to_light_cluster, mass_to_light_field)
    # print(np.c_[cluster_masses, a_true, sigma_center, backgrounds])
    if Reff_error:
        inferred_params = np.array(
            [
                inference_experiment(
                    shape=shape,
                    aperture=100,
                    N=n,
                    res=0.1,
                    count_photons=True,
                    model=model,
                    background=backgrounds[i],
                )
                for i, n in enumerate(N_stars)
            ]
        )
        if model == "EFF":
            Reff_measured = (
                10 ** inferred_params[:, 0]
                * (2 ** (2 / (inferred_params[:, 1] - 2)) - 1) ** 0.5
                / (2 ** (2 / (shape - 2)) - 1) ** 0.5
            ) * Reff_true
        else:
            Reff_measured = a_true * np.array(
                [king62_r50(10**logrc, c) for logrc, c in inferred_params]
            )
            # print(Reff_measured)
    else:
        Reff_measured = Reff_true

    # plt.loglog(cluster_masses, Reff_measured, ".")

    cut = (Reff_measured > 0.1) * (Reff_measured < 1e2)

    print(cut.sum())

    powerlaw_fit_params = np.polyfit(
        np.log10(cluster_masses_measured[cut]), np.log10(Reff_measured[cut]), 1
    )
    print(powerlaw_fit_params)

    return powerlaw_fit_params


sigma_star = 100
N_clusters = 100
mass_error = 0.3
Reff_error = True


def experiment(
    sigma_star=100,
    N_clusters=100,
    mass_error=0.3,
    Reff_error=True,
    Nsamples=100,
    model="King62",
    shape=2.5,
):
    fname = f"powerlaw_fit_params_{model}_N{N_clusters}_sigma{sigma_star}_dm{mass_error}_dR{Reff_error}.dat"

    if not isfile(fname):
        params = Parallel(10)(
            delayed(generate_masses_and_radii)(
                N_clusters=N_clusters,
                sigma_star=sigma_star,
                cluster_mass_error_dex=mass_error,
                Reff_error=True,
                model=model,
                shape=shape,
            )
            for i in range(Nsamples)
        )
        np.savetxt(fname, params)
    else:
        params = np.loadtxt(fname)

    plt.clf()
    fig, ax = plt.subplots()
    mgrid = np.logspace(3, 6, 1000)
    for i, p in enumerate(params):
        ax.loglog(
            mgrid,
            10 ** (p[1] + p[0] * np.log10(mgrid)),
            color="grey",
            alpha=0.3,
            label=("Measured Relations" if i == 0 else None),
        )

    ax.loglog(mgrid, (mgrid / 1e3) ** (1.0 / 3), color="black", label="True Relation")
    ax.set(
        xlabel=r"$M\,\left(M_\odot\right)$",
        ylabel=r"$R_{\rm eff}\,\left(\rm pc\right)$",
    )
    ax.legend()
    plt.savefig(fname.replace(".dat", ".pdf"), bbox_inches="tight")


experiment(1000, 1000, 0.3, True, 100, shape=2.5, model=model)
experiment(1000, 1000, 0.0, True, 100, shape=2.5, model=model)
experiment(1000, 1000, 0.3, False, 100, shape=2.5, model=model)

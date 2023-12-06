"""Plots statistical uncertainty of star cluster R_eff measurement as a function of N"""
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from palettable.colorbrewer.qualitative import Dark2_3
from cycler import cycler
from run_experiments import *


Nsamples = 10**2
Ns = 2 ** np.arange(4, int(np.log2(10**6)))


def basic_plot():
    plt.rc(
        "axes",
        prop_cycle=(
            cycler("linestyle", ["solid", "dashed", "dotted"])
            * cycler("color", Dark2_3.mpl_colors)
        ),
    )
    # shape parameter, background, aperture, model
    models = {
        r"EFF ($\gamma=4$)": [4, 1000, 1e-10, "EFF"],
        r"EFF ($\gamma=2.5$)": [2.5, 1000, 1e-10, "EFF"],
        r"EFF ($\gamma=2.1$)": [2.1, 1000, 1e-10, "EFF"],
        r"King ($c=10$)": [10, 1000, 1e-10, "King62"],
        r"King ($c=30$)": [30, 1000, 1e-10, "King62"],
        r"King ($c=100$)": [100, 1000, 1e-10, "King62"],
    }

    fig, ax = plt.subplots(figsize=(4, 4))

    for m, params in models.items():
        shape, ap, bg, model = params

        stds = []
        for N in Ns:
            print(N)
            result = np.array(
                Parallel(16)(
                    delayed(inference_experiment)(
                        shape, bg, ap, N, res=0.1, model=model
                    )
                    for i in range(Nsamples)
                )
            )

            if model == "King62":
                r50 = np.array([king62_r50(10**loga, c) for loga, c in result])
            else:
                r50 = 10 ** result[:, 0] * (2 ** (2 / (result[:, 1] - 2)) - 1) ** 0.5

            r50[np.isnan(r50)] = 1e100
            r50[~np.isfinite(r50)] = 1e100
            stds.append(np.diff(np.percentile(np.log10(r50), [16, 84]))[0] * 0.5)

        ax.loglog(Ns, stds, label=m, ls=("solid" if model == "EFF" else "dashed"))
    ax.set(
        xlabel=r"$N_{\rm eff}$",
        ylabel=r"$\sigma\left(\hat{R}_{\rm eff}/R_{\rm eff}\right)$ (dex)",
        xlim=[100, 10**6],
        ylim=[1e-3, 1],
    )
    ax.legend(labelspacing=0, loc=3)
    plt.savefig("N_vs_sigmaReff.pdf", bbox_inches="tight")


def photometry_plot():
    # shape parameter, background, aperture, model
    models = {
        #        r"EFF ($\gamma=2.1$)": [2.1, 1000, 1e-10, "EFF"],
        r"EFF ($\gamma=2.5$)": [2.5, 1000, 1e-10, "EFF"],
        #        r"EFF ($\gamma=4$)": [4, 1000, 1e-10, "EFF"],
        r"King ($c=30$)": [30, 1000, 1e-10, "King62"],
        #        r"King ($c=3$)": [10, 1000, 1e-10, "King62"],
        #        r"King ($c=30$)": [30, 1000, 1e-10, "King62"],
    }

    # Ns = 2 ** np.arange(5, int(np.log2(10**6) + 2))

    fig, ax = plt.subplots(figsize=(4, 4))
    for count_photons in False, True:
        for m, params in models.items():
            shape, ap, bg, model = params
            stds = []
            for N in Ns:
                #                print(N)
                result = np.array(
                    Parallel(16)(
                        delayed(inference_experiment)(
                            shape,
                            bg,
                            ap,
                            N,
                            res=0.1,
                            model=model,
                            count_photons=count_photons,
                        )
                        for i in range(Nsamples)
                    )
                )

                if model == "King62":
                    r50 = np.array([king62_r50(10**loga, c) for loga, c in result])
                else:
                    r50 = (
                        10 ** result[:, 0] * (2 ** (2 / (result[:, 1] - 2)) - 1) ** 0.5
                    )

                r50[np.isnan(r50)] = 1e100
                r50[~np.isfinite(r50)] = 1e100
                stds.append(np.diff(np.percentile(np.log10(r50), [16, 84]))[0] * 0.5)

            if count_photons:
                label = m + " (Photometry)"
                color = "red"
            else:
                label = m + " (Star counts)"
                color = "black"

            ax.loglog(
                Ns,
                stds,
                label=label,
                color=color,
                ls=("dashed" if model == "King62" else "solid"),
            )

    handles, labels = ax.get_legend_handles_labels()
    l1 = ax.legend(handles[:2], labels[:2], labelspacing=0, loc=3)
    l2 = ax.legend(handles[2:], labels[2:], labelspacing=0, loc=1)
    ax.add_artist(l1)
    ax.set(
        xlabel=r"$N$",
        ylabel=r"$\sigma\left(\hat{R}_{\rm eff}/R_{\rm eff}\right)$ (dex)",
        xlim=[1000, 10**6],
        ylim=[1e-3, 1],
    )
    plt.savefig("N_vs_sigmaReff_photometry.pdf", bbox_inches="tight")


def background_plot():
    for count_photons in True, False:
        plt.rc(
            "axes",
            prop_cycle=(
                cycler("linestyle", ["solid", "dashed", "dotted"])
                * cycler("color", Dark2_3.mpl_colors)
            ),
        )
        # shape parameter, background, aperture, model
        models = {
            r"EFF ($\gamma=2.5$)": [2.5, 100, 1e-10, "EFF"],
            r"EFF ($\gamma=2.5$) + $10^{-2}$ Background": [2.5, 100, 1e-2, "EFF"],
            r"EFF ($\gamma=2.5$) + $10^{-1}$ Background": [2.5, 100, 1e-1, "EFF"],
            r"King ($c=30$)": [30, 100, 1e-10, "King62"],
            r"King ($c=30$) + $10^{-2}$ Background": [30, 100, 1e-2, "King62"],
            r"King ($c=30$) + $10^{-1}$ Background": [30, 100, 1e-1, "King62"],
        }

        # Ns = 2 ** np.arange(5, int(np.log2(10**6) + 2))

        fig, ax = plt.subplots(figsize=(4, 4))

        for m, params in models.items():
            shape, ap, bg, model = params

            stds = []
            for N in Ns:
                print(N)
                result = np.array(
                    Parallel(4)(
                        delayed(inference_experiment)(
                            shape,
                            bg,
                            ap,
                            N,
                            res=0.1,
                            model=model,
                            count_photons=count_photons,
                        )
                        for i in range(Nsamples)
                    )
                )

                if model == "King62":
                    r50 = np.array([king62_r50(10**loga, c) for loga, c in result])
                else:
                    r50 = (
                        10 ** result[:, 0] * (2 ** (2 / (result[:, 1] - 2)) - 1) ** 0.5
                    )

                r50[np.isnan(r50)] = 1e100
                r50[~np.isfinite(r50)] = 1e100
                stds.append(np.diff(np.percentile(np.log10(r50), [16, 84]))[0] * 0.5)

            ax.loglog(Ns, stds, label=m, ls=("solid" if model == "EFF" else "dashed"))
        ax.set(
            xlabel=r"$N$",
            ylabel=r"$\sigma\left(\hat{R}_{\rm eff}/R_{\rm eff}\right)$ (dex)",
            xlim=[100, 10**6],
            ylim=[1e-3, 1],
        )
        handles, labels = ax.get_legend_handles_labels()
        l1 = ax.legend(handles[:3], labels[:3], labelspacing=0, loc=1)
        l2 = ax.legend(handles[3:], labels[3:], labelspacing=0, loc=3)
        ax.add_artist(l1)
        plt.savefig(
            f"N_vs_sigmaReff_background_"
            + ("light" if count_photons else "star_counts")
            + ".pdf",
            bbox_inches="tight",
        )


def background_aperture_plot():
    for count_photons in True, False:
        plt.rc(
            "axes",
            prop_cycle=(
                cycler("linestyle", ["solid", "dashed", "dotted"])
                * cycler("color", Dark2_3.mpl_colors)
            ),
        )
        # shape parameter, background, aperture, model
        models = {
            # r"EFF ($\gamma=2.5$)": [2.5, 100, 1e-10, "EFF"],
            r"EFF ($\gamma=2.5$) + $10^{-2}$ Background": [2.5, 100, 1e-2, "EFF"],
            r"EFF ($\gamma=2.5$) + $10^{-2}$ Background, Aperture=3": [
                2.5,
                3,
                1e-2,
                "EFF",
            ],
            r"EFF ($\gamma=2.5$) + $10^{-2}$ Background, Aperture=10": [
                2.5,
                10,
                1e-2,
                "EFF",
            ],
        }

        # Ns = 2 ** np.arange(5, int(np.log2(10**6) + 2))

        fig, ax = plt.subplots(figsize=(4, 4))

        for m, params in models.items():
            shape, ap, bg, model = params

            stds = []
            for N in Ns:
                print(N)
                result = np.array(
                    Parallel(4)(
                        delayed(inference_experiment)(
                            shape,
                            bg,
                            ap,
                            N,
                            res=0.1,
                            model=model,
                            count_photons=count_photons,
                        )
                        for i in range(Nsamples)
                    )
                )

                if model == "King62":
                    r50 = np.array([king62_r50(10**loga, c) for loga, c in result])
                else:
                    r50 = (
                        10 ** result[:, 0] * (2 ** (2 / (result[:, 1] - 2)) - 1) ** 0.5
                    )

                r50[np.isnan(r50)] = 1e100
                r50[~np.isfinite(r50)] = 1e100
                stds.append(np.diff(np.percentile(np.log10(r50), [16, 84]))[0] * 0.5)

            ax.loglog(Ns, stds, label=m, ls=("solid" if model == "EFF" else "dashed"))
        ax.set(
            xlabel=r"$N$",
            ylabel=r"$\sigma\left(\hat{R}_{\rm eff}/R_{\rm eff}\right)$ (dex)",
            xlim=[100, 10**6],
            # ylim=[1e-3, 1],
        )
        ax.legend(loc=3)

        plt.savefig(
            f"N_vs_sigmaReff_background_aperture_"
            + ("light" if count_photons else "star_counts")
            + ".pdf",
            bbox_inches="tight",
        )


def main():
    # basic_plot()
    # photometry_plot()
    # background_plot()
    background_aperture_plot()


if __name__ == "__main__":
    main()

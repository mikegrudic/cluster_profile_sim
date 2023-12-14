import pickle
from glob import glob
import numpy as np
from os import system
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sys import argv
from matplotlib import pyplot as plt
from run_experiments import *

bootstrap = int(argv[1])
model = argv[2]


def fit_results(do_plot=False, model="King62"):
    stds = []
    Neffs = []
    N = []
    backgrounds = []
    gammas = []
    bias = []
    stars = []
    apertures = []
    ages = []
    frac_inf = []
    for r in glob(f"results_{model}/*.dump"):
        try:
            p, result = pickle.load(open(r, "rb"))
        except:
            system("rm " + r)
            continue
        if len(result) < 10**3:
            system("rm " + r)
            continue
        if model == "EFF":
            r50 = 10 ** result[:, 0] * np.sqrt(
                2 ** (2 / (np.array(result[:, 1]) - 2)) - 1
            )
            r50_0 = (2 ** (2 / (p[0] - 2)) - 1) ** 0.5
        else:
            r50 = np.array([king62_r50(10**loga, c) for loga, c in result])
            r50_0 = king62_r50(1.0, p[0])
        r50[np.isnan(r50)] = np.inf
        Neff = p[3]
        gammas.append(p[0])
        background = p[1]
        stars.append(p[-1])
        apertures.append(p[2])
        ages.append((p[-1] if len(p) > 6 else 1e8))
        N.append(p[3])
        std = (
            0.5 * np.diff(np.percentile(np.log10(r50 / r50_0), [16, 84]))[0]
        )  # 0.5*np.diff(np.percentile(np.log10(r50 / r50_0), [16, 84]))[0]
        backgrounds.append(background)
        stds.append(std)
        frac_inf.append(1 - np.isfinite(r50).sum() / len(r50))
        bias.append(np.median(np.log10(r50 / r50_0)[np.isfinite(r50)]))

    gammas, ages, stds, N, backgrounds, bias, stars, apertures, frac_inf = (
        np.array(gammas),
        np.array(ages),
        np.array(stds),
        np.array(N),
        np.array(backgrounds),
        np.array(bias),
        np.array(stars),
        np.array(apertures),
        np.array(frac_inf),
    )
    if model == "EFF":
        r50_0 = (2 ** (2 / (gammas - 2)) - 1) ** 0.5
    else:
        r50_0 = np.array([king62_r50(1.0, g) for g in gammas])

    np.save(
        "std_vs_frac.npy",
        np.c_[gammas, ages, stds, N, backgrounds, bias, stars, apertures, frac_inf],
    )
    cut = np.isfinite(stds) * (stds < 0.1) * (ages == 1e8)  # * (stars)
    if bootstrap:
        cut *= np.random.rand(len(stds)) > 0.5

    def std_model(x):
        Reff_model = (2 ** (2 / (gammas[cut] - 2)) - 1) ** 0.5
        Reff_SNR = 0.5 / (
            backgrounds[cut] * np.pi * Reff_model**2
        )  # ratio of true counts within Reff to total counts
        f_background = backgrounds[cut] ** (
            (gammas[cut] - 2) / gammas[cut]
        )  # fraction of counts above the background
        #        print(np.median(f_background))
        ap_over_Reff = apertures[cut] / gammas[cut]  # Reff_model

        bg_threshold = 10 ** x[1] * np.ones(len(Reff_model))
        if len(x) > 8:
            if np.any(stars[cut]):
                bg_threshold[stars[cut]] *= 10 ** x[8]
        m = (
            10 ** x[0]
            * N[cut] ** -0.5
            * (1 + (backgrounds[cut] / bg_threshold) ** x[2])
            * (1 + (apertures[cut] / 10 ** x[3]) ** x[4])
            * (1 + 10 ** x[5] * (gammas[cut] - 2) ** x[6])
            # * (
            #     1
            #     + 10 ** x[5]
            #     * np.square(
            #         2 ** (2 / (gammas[cut] - 2))
            #         / (2 ** (2 / (gammas[cut] - 2)) - 1)
            #         * (gammas[cut] - 2) ** -2
            #     )
            # )
            # x[6]
        )

        if len(x) > 7:
            m[stars[cut]] *= 10 ** x[7]
        return m

    def lossfunc(x, *args):
        m = std_model(x)
        return np.mean(np.log10(m / stds[cut]) ** 2) ** 0.5

    if np.any(stars[cut]) and np.any(~stars[cut]):
        # fit that accounts for both light profiles and star counts
        sol = minimize(
            lossfunc,
            (0.0, 0, 0.5, 0, 0.5, 0, -1, 1, 0),
            tol=1e-6,
        )
    else:
        sol = minimize(
            lossfunc,
            (0.0, 0, 0.5, 0, 0.5, 0, -1),
            tol=1e-6,
        )
    # sol = minimize(lossfunc, (0.0, 0, 0.5, 0, 0.5, 0, -1, 1), tol=1e-6)
    print(sol.fun)

    if do_plot == 1:
        cut = N > 0
        fig, ax = plt.subplots()
        Neff = (std_model(sol.x) / 10 ** sol.x[0]) ** -2
        # print(
        #     Neff[N.argmax()],
        #     N.max(),
        #     stars[N.argmax()],
        #     std_model(sol.x)[N.argmax()],
        #     (apertures / r50_0)[N.argmax()],
        #     backgrounds[N.argmax()],
        # )
        Ngrid = np.logspace(-1, 6, 1000)

        # ax.set(xscale="log", yscale="log")
        # ax.plot(
        #     stds[cut],
        #     std_model(sol.x),
        #     ls="dashed",
        #     color="black",
        #     zorder=-100,
        #     label=r"$%3.2g/\sqrt{N_{\rm eff}}$" % (10 ** sol.x[0]),
        # )
        print(len(gammas), len(stars), len(stds), len(std_model(sol.x)))
        ax.scatter(
            gammas[stars],
            std_model(sol.x)[stars] / stds[stars],
            s=6,
            label="Fit surface brightness",
            marker="o",
            color="black",
            lw=0.1,
        )
        ax.scatter(
            gammas[cut][~stars],
            std_model(sol.x)[~stars] / stds[~stars],
            s=6,
            label="Fit number density",
            marker="s",
            edgecolor="black",
            lw=0.1,
            facecolor=None,
            color=(0, 0, 0, 0),
        )
        ax.set(
            # xscale="log",
            yscale="log",
            # xlim=[1e-4, 1],
            # ylim=[1e-4, 1],
            # ylim=[10**-4, 1],
            # xlabel=r"$N_{\rm eff}$",
            ylabel=r"$\sigma\left(\hat{R}_{\rm eff}/R_{\rm eff,true}\right)\,\left(\rm dex\right)$",
        )
        ax.legend(frameon=True, edgecolor="black")
        plt.savefig("sigma_vs_sigmamodel.pdf", bbox_inches="tight")

    elif do_plot == 2:
        cut = N > 0
        fig, ax = plt.subplots()
        Neff = (model(sol.x) / 10 ** sol.x[0]) ** -2
        ax.scatter(Neff, bias)
        ax.set(xscale="log")  # , yscale="log")
        plt.show()
    return sol.x


if bootstrap:
    results = np.array([fit_results(model=model) for i in range(bootstrap)])
    for i in 0, 1, 3, 5, 8:
        if i < len(results):
            results[:, i] = 10 ** results[:, i]
    sigma = np.diff(np.percentile(results, [16, 84], axis=0), axis=0)[0]
    print(np.c_[np.median(results, axis=0), sigma])
    bootstrap = 0

results = fit_results(do_plot=1, model=model)
for i in 0, 1, 3, 5, 8:
    if i < len(results):
        results[i] = 10 ** results[i]
print(results)

# %%
import pickle
from glob import glob
import numpy as np
from os import system
from scipy.optimize import minimize
from matplotlib import pyplot as plt

stds = []
Neffs = []
N = []
backgrounds = []
gammas = []
bias = []
stars = []
apertures = []
ages = []
for r in glob("results/*.dump"):
    try:
        p, result = pickle.load(open(r, "rb"))
    except:
        system("rm " + r)
        continue
    if len(result) < 10**4:
        system("rm " + r)
        continue
    r50 = 10 ** result[:, 0] * np.sqrt(2 ** (2 / (np.array(result[:, 1]) - 2)) - 1)
    r50_0 = (2 ** (2 / (p[0] - 2)) - 1) ** 0.5
    r50[np.isnan(r50)] = np.inf
    Neff = p[3]
    gammas.append(p[0])
    background = p[1]
    stars.append(p[-1])
    apertures.append(p[2])
    ages.append((p[-1] if len(p) > 6 else 1e8))
    N.append(p[3])
    std = np.diff(np.percentile(np.log10(r50 / r50_0), [16, 84]))[0]
    backgrounds.append(background)
    stds.append(std)
    bias.append(np.median(np.log10(r50 / r50_0)[np.isfinite(r50)]))

gammas, ages, stds, N, backgrounds, bias, stars, apertures = (
    np.array(gammas),
    np.array(ages),
    np.array(stds),
    np.array(N),
    np.array(backgrounds),
    np.array(bias),
    np.array(stars),
    np.array(apertures),
)
cut = np.isfinite(stds) * (stds < 0.1) * (ages == 1e8)  # * (backgrounds==1e-6)
cut *= np.random.rand(len(stds)) > 0.5


def model(x):
    Reff_model = (2 ** (2 / (gammas[cut] - 2)) - 1) ** 0.5
    Reff_SNR = 0.5 / (
        backgrounds[cut] * np.pi * Reff_model**2
    )  # ratio of true counts within Reff to total counts
    ap_over_Reff = apertures[cut] / Reff_model

    m = (
        10 ** x[0]
        * N[cut] ** -0.5
        # * (1 + (Reff_SNR / 10 ** x[1]) ** x[2])
        * (1 + (backgrounds[cut] / 10 ** x[1]) ** x[2])
        * (1 + (ap_over_Reff / 10 ** x[3]) ** x[4])
        * (1 + 10 ** x[6] * (gammas[cut] - 2) ** x[7])
    )

    if np.any(stars[cut]):
        m[stars[cut]] *= 10 ** x[5]
    return m


def lossfunc(x, *args):
    m = model(x)
    return np.mean(np.log(m / stds[cut]) ** 2) ** 0.5


sol = minimize(lossfunc, (0.0, 0, 0.5, 0, 0.5, 1, 0, -1), tol=1e-6)
# sol = minimize(lossfunc, (0.0, 0, 0.5, 1, 0, -1), tol=1e-6)
# sol = minimize(lossfunc, (0.0, 1, 0, -1), tol=1e-6)
# print(sol)
# print(sol.fun)

error = np.sqrt(sol.fun * np.diag(sol.hess_inv))
# print(np.linalg.eigh(sol.hess_inv)[1][:, -1])
# print(sol.jac)
print(sol.success)
print(np.c_[sol.x, 10**sol.x])

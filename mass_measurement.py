"""Experiments with effect of cluster radius uncertainties on mass-radius relation"""
import numpy as np
from star_cluster import *
from filterlists import messa_m51_filters
from matplotlib import pyplot as plt


def mock_measure_radii(
    N_clusters=100,
    Mmin=10**3,
    Mmax=10**7,
):
    Mmin, Mmax = 1e3, 10**7
    x = np.random.rand(N_clusters)
    cluster_masses = Mmax * Mmin / (Mmax * (1 - x) + Mmin * x)
    cluster_ages = 1e6 * 10 ** (3 * np.random.rand(N_clusters))
    cluster_N = np.int_(cluster_masses / 0.5 + 0.5)

    cs = cluster_slug(use_nebular=False, photsystem="Vega")
    cs.add_filters(messa_m51_filters)

    mass_measurements = []
    age_measurements = []
    for i in range(N_clusters):
        print(i)
        mlow, mmed, mhi = StarCluster(cluster_N[i]).measure_slug(
            cluster_ages[i], "Mass", cs=cs, filters=messa_m51_filters
        )
        tlow, tmed, thi = StarCluster(cluster_N[i]).measure_slug(
            cluster_ages[i], "Age", cs=cs, filters=messa_m51_filters
        )
        age_measurements.append([tlow, tmed, thi])
        mass_measurements.append([mlow, mmed, mhi])
    return (
        cluster_masses,
        cluster_ages,
        np.array(mass_measurements),
        np.array(age_measurements),
    )


m, age, logmmeas, logtmeas = mock_measure_radii(1000)
np.save("mass_measurements.npy", np.c_[m, logmmeas, age, logtmeas])
print(np.std(np.log10(m) - logmmeas[:, 1]))
plt.scatter(np.log10(m), logmmeas[:, 1], c=np.log10(age))
plt.show()

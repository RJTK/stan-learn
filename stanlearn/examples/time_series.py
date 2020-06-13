import seaborn
import os
import matplotlib.pyplot as plt

import numpy as np
from numpy.polynomial.polynomial import polyroots

from stanlearn.time_series import BayesAR

try:
    FIGURE_DIR = os.path.join(os.path.dirname(__file__),
                              "./figures/")
except NameError:  # no __file__ when interactive
    FIGURE_DIR = "./figures/"

try:
    os.mkdir(FIGURE_DIR)
except FileExistsError:
    pass


def basic_example():
    p = 3  # Misspecify p
    T = 100
    v = 0.25 * np.random.normal(size=T + p)
    y = np.array(v)
    b1 = 0.6
    b2 = -0.8

    mu = -0.7
    r = 1.2 / T

    true_roots = polyroots(np.append(
        -np.array([b1, b2] + [0.0] * (p - 2))[::-1], 1))

    for t in range(T):
        y[t] = b1 * y[t - 1] + b2 * y[t - 2] + v[t]
    y = y + mu + r * np.arange(-p, T)

    y = y[p:].reshape(-1, 1)

    ar = BayesAR(normalize=False, p=p)
    ar.fit(y)

    fig, _ = ar.plot_ppc(y, show=False)
    fig.savefig(FIGURE_DIR + "time_series_ppc.png")
    fig.savefig(FIGURE_DIR + "time_series_ppc.pdf")
    plt.show()

    fig, axes = ar.plot_posterior_params(show=False)
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    fig.savefig(FIGURE_DIR + "time_series_param_posterior.png")
    fig.savefig(FIGURE_DIR + "time_series_param_posterior.pdf")
    plt.show()

    return


if __name__ == "__main__":
    basic_example()

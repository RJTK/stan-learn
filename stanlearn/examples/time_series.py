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
    ar = BayesAR(normalize=False, p=p)
    T = 100
    v = 0.25 * np.random.normal(size=T)
    y = np.array(v)
    b1 = 0.6
    b2 = -0.8

    for t in range(2, T):
        y[t] = b1 * y[t - 1] + b2 * y[t - 2] + v[t]

    y = y.reshape(-1, 1)
    ar.fit(y)

    y_ppc = ar.get_ppc()

    plt.plot(y_ppc[::10].T, linewidth=0.5, color="m", alpha=0.25)
    plt.plot(y.ravel(), linewidth=2.0, color="b", alpha=0.8, label="y")
    plt.plot(np.mean(y_ppc, axis=0), linewidth=2.0, color="r",
             alpha=0.8, label="y\_ppc")
    plt.xlabel("$t$")
    plt.ylabel("$y$")
    plt.title("AR(p) model PPC")
    plt.legend()
    plt.savefig(FIGURE_DIR + "time_series_ppc.png")
    plt.savefig(FIGURE_DIR + "time_series_ppc.pdf")
    plt.show()

    true_roots = polyroots(np.append(
        -np.array([b1, b2] + [0.0] * (p - 2))[::-1], 1))

    fig, axes = ar.plot_posterior_params(show=False)
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="r")
    axes[1].legend()
    fig.savefig(FIGURE_DIR + "time_series_param_posterior.png")
    fig.savefig(FIGURE_DIR + "time_series_param_posterior.pdf")
    plt.show()

    return


if __name__ == "__main__":
    basic_example()

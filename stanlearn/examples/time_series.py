import seaborn
import os
import matplotlib.pyplot as plt

import numpy as np
from numpy.polynomial.polynomial import polyroots

from stanlearn.time_series import BayesAR, MixtureBayesAR

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

    ax = ar.plot_ppc(y, show=False)
    fig = ax.figure
    fig.savefig(FIGURE_DIR + "time_series_ppc.png")
    fig.savefig(FIGURE_DIR + "time_series_ppc.pdf")
    plt.show()

    axes = ar.plot_posterior_params(show=False)
    fig = axes[0].figure
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    fig.savefig(FIGURE_DIR + "time_series_param_posterior.png")
    fig.savefig(FIGURE_DIR + "time_series_param_posterior.pdf")
    fig.show()

    return


def mixture_example():
    p_max = 5
    T = 1000
    v = 0.25 * np.random.normal(size=T + p_max)
    y = np.array(v)
    b1 = 0.6
    b2 = -0.8

    mu = -0.7
    r = 1.2 / T

    true_roots = polyroots(np.append(
        -np.array([b1, b2])[::-1], 1))

    for t in range(T):
        y[t] = b1 * y[t - 1] + b2 * y[t - 2] + v[t]
    y = y + mu + r * np.arange(-p_max, T)
    y = y[p_max:].reshape(-1, 1)

    nu_th = 3  # Priors for model order
    mu_th = 1. / np.arange(1, p_max + 2)**(1./3)
    mu_th /= sum(mu_th)

    ar = MixtureBayesAR(normalize=False, p_max=p_max, n_chains=4, warmup=3000,
                        nu_th=nu_th, mu_th=mu_th)
    ar.fit(y)

    ax = ar.plot_ppc(y, show=False)
    fig = ax.figure
    fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.png")
    fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.pdf")
    plt.show()

    fig, axes = plt.subplots(1, 2)
    axes = axes.ravel()
    ar.plot_posterior_params(show=False, ax=axes[0])
    ar.plot_poles(p=None, show=False, ax=axes[1])
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.png")
    fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.pdf")
    plt.show()

    return


if __name__ == "__main__":
    basic_example()
    mixture_example()

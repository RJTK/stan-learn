import os
import matplotlib.pyplot as plt

import numpy as np
from numpy.polynomial.polynomial import polyroots

from stanlearn.time_series import (BayesAR, BayesRepAR,
                                   BayesMixtureAR)

try:
    FIGURE_DIR = os.path.join(os.path.dirname(__file__),
                              "./figures/")
except NameError:  # no __file__ when interactive
    FIGURE_DIR = "./figures/"

try:
    os.mkdir(FIGURE_DIR)
except FileExistsError:
    pass

# TODO: How do I properly configure which c compiler?
os.environ["CXX"] = "/usr/bin/g++"


def basic_example():
    # Repetitions of the same model
    p = 3  # Misspecify p
    T = 100
    sigma = 1.2
    v = sigma * np.random.normal(size=T)
    y = np.array(v)
    b1 = 0.6
    b2 = -0.8

    mu = -0.7
    r = 1.2 / T

    true_roots = polyroots(np.append(
        -np.array([b1, b2] + [0.0] * (p - 2))[::-1], 1))

    for t in range(T):
        y[t] = b1 * y[t - 1] + b2 * y[t - 2] + v[t]
    y = y + mu + r * np.arange(T)

    ar = BayesAR(normalize=False, p=p, warmup=500, samples_per_chain=500)
    ar.fit(y, show_progress=True, output_dir="./stan_output/")

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    ar.plot_ppc(y, show=False, ax=ax)
    # fig.savefig(FIGURE_DIR + "time_series_ppc.png")
    # fig.savefig(FIGURE_DIR + "time_series_ppc.pdf")
    plt.show()

    axes = ar.plot_posterior_params(show=False)
    fig = axes[0].figure
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    # fig.savefig(FIGURE_DIR + "time_series_param_posterior.png")
    # fig.savefig(FIGURE_DIR + "time_series_param_posterior.pdf")
    fig.show()

    ax = ar.plot_ll_trace(show=True)
    return


def ar_rep_example():
    # Multiple repetitions
    p = 3  # Misspecify p
    T = 150
    K = 5
    sigma = 0.5 + np.random.uniform(size=K)**2
    v = sigma[None, :] * np.random.normal(size=(T, K))
    y = np.array(v)
    b1 = 0.6
    b2 = -0.8

    # mu = -0.7
    # r = 1.2 / T
    mu = -0.7
    r = 1.2 / T

    true_roots = polyroots(np.append(
        -np.array([b1, b2] + [0.0] * (p - 2))[::-1], 1))

    for t in range(T):
        y[t, :] = b1 * y[t - 1, :] + b2 * y[t - 2, :] + v[t, :]
    y = y + mu + r * np.arange(T)[:, None]

    ar = BayesRepAR(normalize=False, p=p, warmup=1000,
                    samples_per_chain=1000)
    ar.fit(y)

    fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
    for k in range(1, 5):
        ar.plot_ppc(y, k=k, show=False, ax=ax[k - 1],
                    labels=(k == 3))
        ax[k - 1].set_title(f"$k = {k}$")
    fig = ax[0].figure
    fig.suptitle("$AR(p)$ PPC")
    # fig.savefig(FIGURE_DIR + "time_series_ppc.png")
    # fig.savefig(FIGURE_DIR + "time_series_ppc.pdf")
    plt.show()

    # Posterior for particular k
    axes = ar.plot_posterior_params(show=False, k=2)
    fig = axes[0].figure
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    # fig.savefig(FIGURE_DIR + "time_series_param_posterior.png")
    # fig.savefig(FIGURE_DIR + "time_series_param_posterior.pdf")
    fig.show()

    # Posterior for the hierarchical params
    axes = ar.plot_posterior_params(k=None)
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    plt.show()

    ar.plot_ll_trace(show=True)
    return


def mixture_example1():
    p_max = 5
    T = 50
    v = 0.25 * np.random.normal(size=T + p_max)
    y = np.array(v)
    b1 = 1.6
    b2 = -0.8

    mu = -0.7
    r = 1.2 / T

    true_roots = polyroots(np.append(
        -np.array([b1, b2])[::-1], 1))

    for t in range(T):
        y[t] = b1 * y[t - 1] + b2 * y[t - 2] + v[t]
    y = y + mu + r * np.arange(-p_max, T)

    nu_th = 3  # Priors for model order
    mu_th = 1. / np.arange(1, p_max + 2)**(1./3)
    mu_th /= sum(mu_th)

    ar = BayesMixtureAR(normalize=False, p_max=p_max, n_chains=4, warmup=3000,
                        nu_th=nu_th, mu_th=mu_th)
    ar.fit(y)

    ax = ar.plot_ppc(y, show=False)
    fig = ax.figure
    fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.png")
    fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.pdf")
    plt.show()

    axes = ar.plot_posterior_params(show=False)
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.png")
    fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.pdf")
    plt.show()

    return

def mixture_example2():
    p_max = 5
    T = 1000
    v = 0.25 * np.random.normal(size=T + p_max)
    y = np.array(v)

    mu = -0.7
    r = 1.2 / T

    b1 = 0.9
    b2 = 0.5
    b3 = -0.8

    true_roots = polyroots(np.append(
        -np.array([b1, b2, b3])[::-1], 1))
    print(np.abs(true_roots))

    for t in range(T):
        y[t] = b1 * y[t - 1] + b2 * y[t - 2] + b3 * y[t - 3] + v[t]
    y = y + mu + r * np.arange(T)
    y = y[p_max:].reshape(-1, 1)

    mu_th = np.ones(p_max + 1)  # / np.arange(1, p_max + 2)**(1./3)
    mu_th /= sum(mu_th)

    ar = BayesMixtureAR(normalize=False, p_max=p_max, n_chains=4, warmup=3000,
                        mu_th=mu_th)
    ar.fit(y)

    ax = ar.plot_ppc(y, show=False)
    fig = ax.figure
    # fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.png")
    # fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.pdf")
    plt.show()

    fig, axes = plt.subplots(1, 2)
    axes = axes.ravel()
    ar.plot_posterior_params(show=False, ax=axes[0])
    ar.plot_poles(p=None, show=False, ax=axes[1])
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    # fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.png")
    # fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.pdf")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ar.plot_posterior_params(show=False, ax=ax)
    plt.show()

    return


def mixture_example3():
    p_max = 5
    T = 1000
    v = 0.25 * np.random.normal(size=T + p_max)
    y = np.array(v)

    mu = -0.7
    r = 1.2 / T

    b1 = 0.9
    b2 = -0.5
    b3 = 0.0
    b4 = 0.6
    b5 = -0.5

    true_roots = polyroots(np.append(
        -np.array([b1, b2, b3, b4, b5])[::-1], 1))
    print(np.abs(true_roots))

    for t in range(T):
        y[t] = (b1 * y[t - 1] + b2 * y[t - 2] + b3 * y[t - 3] +
                b4 * y[t - 4] + b5 * y[t - 5] + v[t])
    y = y + mu + r * np.arange(-p_max, T)
    y = y[p_max:].reshape(-1, 1)

    mu_th = np.ones(p_max + 1)  # / np.arange(1, p_max + 2)**(1./3)
    mu_th /= sum(mu_th)

    ar = BayesMixtureAR(normalize=False, p_max=p_max, n_chains=4, warmup=3000,
                        mu_th=mu_th)
    ar.fit(y)

    ax = ar.plot_ppc(y, show=False)
    fig = ax.figure
    # fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.png")
    # fig.savefig(FIGURE_DIR + "mixture_time_series_ppc.pdf")
    plt.show()

    fig, axes = plt.subplots(1, 2)
    axes = axes.ravel()
    ar.plot_posterior_params(show=False, ax=axes[0])
    ar.plot_poles(p=None, show=False, ax=axes[1])
    axes[1].scatter(true_roots.real, true_roots.imag, marker="o",
                    label="True Poles", color="#117733")
    axes[1].legend(loc="upper right")
    # fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.png")
    # fig.savefig(FIGURE_DIR + "mixture_time_series_param_posterior.pdf")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ar.plot_posterior_params(show=False, ax=ax)
    plt.show()

    return


if __name__ == "__main__":
    basic_example()
    mixture_example1()

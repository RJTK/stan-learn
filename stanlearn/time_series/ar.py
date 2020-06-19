import os
from itertools import chain

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyroots

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler

import arviz

from stanlearn.base import StanCacheMixin

try:
    MODEL_DIR = os.path.join(os.path.dirname(__file__),
                             "./stan_models/")
except NameError:  # no __file__ when interactive
    MODEL_DIR = "./stan_models/"


def _compute_roots(b):
    roots = []
    for bi in b:
        roots.append(polyroots(np.append(-bi[::-1], 1)))
    return np.vstack(roots)


# This is basically an abstract base class -- I'm not using formal mechanisms
class BaseAR(BaseEstimator, RegressorMixin, StanCacheMixin):
    def __init__(self, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True):
        BaseEstimator.__init__(self)
        StanCacheMixin.__init__(self, MODEL_DIR)

        self.stan_model, self.predict_model = self._load_compiled_models()

        self.stan_fitting_kwargs = {"chains": n_chains,
                                    "iter": samples_per_chain + warmup,
                                    "warmup": warmup, "init": "random",
                                    "init_r": 1.0, "n_jobs": n_jobs,
                                    "control": {"metric": "diag_e",
                                                "adapt_delta": 0.9}}

        self._fit_results = None
        self.normalize = normalize
        self.p = None

        # For transformations
        self._mean = 0
        self._scale = 0
        return

    def fit(self, data, stan_fitting_kwargs, pars):
        if self.normalize:
            # Don't scale columns independently
            # I want to keep the relative scales
            self._mean = np.mean(data["y"])
            self._scale = np.std(data["y"])
            data["y"] = (data["y"] - self._mean) / self._scale

        fit_kwargs = self._setup_predict_kwargs(data, stan_fitting_kwargs)
        self._fit_results = self.stan_model.sampling(**fit_kwargs)
        print(self._fit_results.stansummary(pars))
        return

    def get_ppc(self):
        """
        A built in PPC for every fit.
        """
        y_ppc = self._fit_results.extract("y_ppc")["y_ppc"]
        if self.normalize:
            y_ppc = self._scale * (y_ppc + self._mean)
        return y_ppc

    def get_trend(self):
        y_trend = self._fit_results.extract("trend")["trend"]
        if self.normalize:
            y_trend = self._scale * (y_trend + self._mean)
        return y_trend

    def plot_ll_trace(self, ax=None, show=False):
        """
        This function can hopefully help to diagnose obvious
        sampling problems like a chain getting stuck.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        y_ll = self._fit_results.extract("y_ll", permuted=False)["y_ll"]
        if len(y_ll.shape) > 2:
            y_ll = np.sum(y_ll, axis=2)

        y_ll = y_ll - np.mean(y_ll)
        t = np.arange(y_ll.shape[0] * y_ll.shape[1])
        t = t.reshape(y_ll.shape[1], y_ll.shape[0]).T
        for i in range(y_ll.shape[1]):
            ax.plot(t[:, i], y_ll[:, i], alpha=0.75, label=f"Chain {i}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Model Log-Likelihood")
        ax.set_title("Log-Likelihood Traceplot")
        ax.legend()
        if show:
            plt.show()
        return ax

    def plot_ppc(self, y, y_ppc, y_trend, ax=None, show=False, labels=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(y_trend.T, linewidth=0.5, color="#88CCEE", alpha=0.1)
        ax.plot(y_ppc.T, linewidth=0.5, color="#CC6677", alpha=0.1)
        ax.plot(y.ravel(), linewidth=2.0, color="#117733", alpha=0.8,
                label="y")
        ax.plot(np.mean(y_ppc, axis=0), linewidth=2.0, color="#882255",
                alpha=0.8, label="y\_ppc")
        ax.plot([], [], linewidth=2, color="#88CCEE", label="trend")

        if labels:
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.set_title("AR(p) model PPC")
            ax.legend(loc="upper right")

        if show:
            plt.show()
        return ax

    def plot_posterior_basic(self, param_df, k, ax, g_proto="g[{tau}]",
                             other_param_names=None):
        """
        Plot AR(p) parameter posteriors.

        param_df should contain ["g", "sigma", "mu", "r"].
        """
        if other_param_names is None:
            other_param_names = ["sigma", "mu", "r"]

        p = self.p
        param_df = pd.DataFrame(param_df)  # Copy the input

        g_params = [g_proto.format(tau=tau) for tau in range(1, p + 1)]
        params = (other_param_names + g_params)

        g_params_tex = [f"$\\Gamma_{{{tau}}}$"for tau in range(1, p + 1)]
        names = (["$\\sigma^2$", "$\\mu_y$", "$r$"] + g_params_tex)
        rename = {frm: to for frm, to in zip(params, names)}

        param_df = param_df.loc[:, params]\
            .rename(rename, axis=1)
        param_df.loc[:, "$\\sigma^2$"] = param_df.loc[:, "$\\sigma^2$"]**2
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=300)
        ax.set_title(f"Parameter Posteriors")

        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax)
        return ax

    def plot_posterior_hier(self, param_df, ax):
        """
        Plot the hierarchical params of a K-repeated AR(p)
        model.

        param_df should contain ["g_hier", "sigma_hier",
                                 "nu_g", "nu_sigma", "lambda"]
        """
        p = self.p
        param_df = pd.DataFrame(param_df)  # Copy the input

        g_params = [f"g_hier[{tau}]"for tau in range(1, p + 1)]
        params = (["sigma_hier", "nu_g", "nu_sigma", "lambda"]
                  + g_params)

        g_params_tex = [f"$\\bar{{\\Gamma}}_{{{tau}}}$"
                        for tau in range(1, p + 1)]
        names = (["$\\bar{\\sigma}^2$", "$\\mathrm{log}_{10}(\\nu_\\Gamma)$",
                  "$\\mathrm{log}_{10}(\\nu_\\sigma)$",
                  "$\\bar{\\lambda}$"]
                 + g_params_tex)
        rename = {frm: to for frm, to in zip(params, names)}

        param_df.loc[:, "sigma_hier"] =\
            param_df.loc[:, "sigma_hier"]**2
        param_df.loc[:, "nu_g"] = np.log10(param_df.loc[:, "nu_g"])
        param_df.loc[:, "nu_sigma"] = np.log10(param_df.loc[:, "nu_sigma"])

        param_df = param_df.loc[:, params]\
            .rename(rename, axis=1)
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=300)
        ax.set_title(f"Hierarchical Parameter Posteriors")

        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax)
        return

    def plot_roots(self, param_df, ax, b_proto="b[{tau}]",
                   title="System Poles"):
        """
        Plot the poles of an AR(p) model.

        param_df should contain "b".
        """
        p = self.p
        param_df = pd.DataFrame(param_df)  # Copy the input

        b_params = [b_proto.format(tau=tau) for tau in range(1, p + 1)]

        b = param_df.loc[:, b_params].to_numpy()
        roots = _compute_roots(b)

        # Z-plot
        uc = patches.Circle((0, 0), radius=1, fill=False,
                            color='black', linestyle='dashed')
        ax.add_patch(uc)
        ax.scatter(roots.real, roots.imag, color="#882255",
                   marker="x", alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        return ax


class BayesAR(BaseAR):
    def __init__(self, p=1, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True):
        BaseAR.__init__(self, n_jobs=n_jobs, warmup=warmup,
                        samples_per_chain=samples_per_chain,
                        n_chains=4, normalize=True)
        self.p = p
        return

    def fit(self, X, y=None, sample_weight=None, **stan_fitting_kwargs):
        """
        "Fit" the model, that is, sample from the posterior.

        params:
            X (n_examples, m_features): Signal to fit, T x 1
            sample_weight: NotImplemented
            stan_fitting_kwargs: To be passed to pystan's .sampling method
        """
        if sample_weight is not None:
            raise NotImplementedError("sampling weighting is not implemented.")
        T = len(X)

        data = {"T": T, "p": self.p, "y": X}
        pars = ["mu", "r", "sigma", "g", "b"]
        super().fit(data, stan_fitting_kwargs, pars)
        return

    def plot_ppc(self, y, ax=None, show=False, labels=True):
        y_ppc = self.get_ppc()
        y_trend = self.get_trend()

        ax = super().plot_ppc(y, y_ppc, y_trend, ax=ax, show=show,
                              labels=labels)
        return ax

    def plot_posterior_params(self, ax=None, show=False):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        if ax is not None:
            raise NotImplementedError

        param_df = self._fit_results.to_dataframe(["b", "g", "sigma",
                                                   "mu", "r"])
        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()

        # Plot parameters
        # fig.suptitle("$y(t) \\sim \\mathcal{N}(\\mu_y + rt + "
        #              "\\sum_{\\tau = 1}^p b_\\tau y(t - \\tau), \\sigma^2); "
        #              "\\frac{1}{2}(1 + \\Gamma_\\tau) \\sim "
        #              "\\beta_\\mu(\\mu_\\beta, \\nu_\\beta)$")
        fig.suptitle("$\\Gamma$-parameterized AR(p) model with trend.")

        super().plot_posterior_basic(param_df, k=None, ax=ax[0],
                                     g_proto="g[{tau}]")
        super().plot_roots(param_df, ax=ax[1], b_proto="b[{tau}]")

        if show:
            plt.show()
        return ax


class BayesRepAR(BaseAR):
    def __init__(self, p=1, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True):
        BaseAR.__init__(self, n_jobs=n_jobs, warmup=warmup,
                        samples_per_chain=samples_per_chain,
                        n_chains=4, normalize=True)

        self.p = p  # The model order
        return

    def fit(self, X, y=None, sample_weight=None, **stan_fitting_kwargs):
        """
        "Fit" the model, that is, sample from the posterior.

        params:
            X (n_examples, m_features): Signal to fit, T x K
            sample_weight: NotImplemented
            stan_fitting_kwargs: To be passed to pystan's .sampling method
        """
        if sample_weight is not None:
            raise NotImplementedError("sample weighting is not implemented.")

        T, K = X.shape

        data = {"T": T, "p": self.p, "y": X.T, "K": K}
        pars = ["mu", "r", "sigma_hier", "nu_sigma", "sigma",
                "nu_g", "g_beta", "mu_beta", "g", "b", "lambda"]

        super().fit(data, stan_fitting_kwargs, pars)
        return

    def plot_ppc(self, y, k=1, ax=None, show=False, labels=True):
        y_trend = self.get_trend()[:, k - 1, :]
        y_ppc = self.get_ppc()[:, k - 1, :]

        super().plot_ppc(y[:, k - 1], y_ppc, y_trend, ax=ax, show=show,
                         labels=labels)

        if show:
            plt.show()
        return ax

    def plot_posterior_params(self, k=None, ax=None, show=False):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        if ax is not None:
            raise NotImplementedError

        param_df = self._fit_results.to_dataframe(["b", "g", "sigma",
                                                   "nu_g", "mu", "r",
                                                   "b_hier", "g_hier",
                                                   "sigma_hier", "nu_sigma",
                                                   "lambda"])

        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        if k is not None:
            super().plot_posterior_basic(
                param_df, k=k, ax=ax[0], g_proto=f"g[{k},{{tau}}]",
                other_param_names=[f"sigma[{k}]", f"mu[{k}]", f"r[{k}]"])
            super().plot_roots(param_df, ax=ax[1], b_proto=f"b[{k},{{tau}}]")
            fig.suptitle("$\\Gamma$-parameterized $\\mathrm{AR}(p)$ model "
                         f"with trend.  Sample $k = {k}$")
        else:
            super().plot_posterior_hier(param_df, ax=ax[0])
            super().plot_roots(param_df, ax=ax[1], b_proto="b_hier[{tau}]",
                               title="Hierarchy Implied Poles")

        if show:
            plt.show()
        return ax


class BayesMixtureAR(BaseEstimator, RegressorMixin, StanCacheMixin):
    def __init__(self, p_max=1, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True, max_samples_mem=500,
                 mu_th=None, nu_th=2):
        BaseEstimator.__init__(self)
        StanCacheMixin.__init__(self, MODEL_DIR)

        self.p_max = p_max  # The model order
        self.stan_model, self.predict_model = self._load_compiled_models()

        self.stan_fitting_kwargs = {"chains": n_chains,
                                    "iter": samples_per_chain + warmup,
                                    "warmup": warmup, "init": "random",
                                    "init_r": 1.0, "n_jobs": n_jobs,
                                    "control": {"metric": "dense_e",
                                                "adapt_delta": 0.9}}

        # self.nu_th = nu_th
        if mu_th is None:
            mu_th = np.ones(p_max + 1)
        mu_th = mu_th / sum(mu_th)  # Ensure it is a simplex

        # In stan the last index is for the AR(0) model.
        self.mu_th = np.append(mu_th[1:], mu_th[0])

        self._fit_results = None
        self.normalize = normalize
        self.max_samples_mem = max_samples_mem

        if normalize:
            self._X_ss = StandardScaler()
        return

    def fit(self, X, y=None, sample_weight=None, **stan_fitting_kwargs):
        """
        "Fit" the model, that is, sample from the posterior.

        params:
            X (n_examples, m_features): Signal to fit, T x 1
            sample_weight: NotImplemented
            stan_fitting_kwargs: To be passed to pystan's .sampling method
        """
        if sample_weight is not None:
            raise NotImplementedError("sampling weighting is not implemented.")
        T, n = X.shape
        if n > 1:
            raise NotImplementedError

        if self.normalize:
            X = self._X_ss.fit_transform(X)

        X = X.ravel()
        data = {"T": T, "p_max": self.p_max, "y": X,
                "mu_th": self.mu_th}

        fit_kwargs = self._setup_predict_kwargs(data, stan_fitting_kwargs)
        self._fit_results = self.stan_model.sampling(**fit_kwargs)

        def iter_tau(name):
            return [f"{name}[{tau}]" for tau in range(1, self.p_max + 1)]

        pars = ["mu", "r", "sigma", "nu_gamma", "gamma", "theta"]

        print(self._fit_results.stansummary(pars))
        return

    def predict(self, X, ret_posterior=False, **stan_fitting_kwargs):
        """
        Produce samples from the predictive distribution.  This can be
        used for either prior predictive checks or for posterior
        predictions.

        params:
            X (n_examples, m_features): Regressors
            ret_posterior: Whether or not to return all the
                posterior samples.  If false, we only return the
                posterior mean, which is dramatically faster.
            stan_fitting_kwargs: kwargs for pystan's sampling method.

        returns:
            y_hat (n_examples), y_samples (k_samples, n_examples) -- (if
                ret_posterior=True)
            y_hat (n_examples) -- (otherwise)
        """
        raise NotImplementedError

    def get_ppc(self):
        """
        A built in PPC for every fit.
        """
        return self._fit_results.extract("y_ppc")["y_ppc"]

    def get_trend(self):
        return self._fit_results.extract("trend")["trend"]

    def plot_ppc(self, y, show=False):
        fig, ax = plt.subplots(1, 1)
        y_trend = self.get_trend()
        y_ppc = self.get_ppc()

        ax.plot(y_trend.T, linewidth=0.5, color="#88CCEE", alpha=0.1)
        ax.plot(y_ppc.T, linewidth=0.5, color="#CC6677", alpha=0.1)
        ax.plot(y.ravel(), linewidth=2.0, color="#117733", alpha=0.8,
                label="y")
        ax.plot(np.mean(y_ppc, axis=0), linewidth=2.0, color="#882255",
                alpha=0.8, label="y\_ppc")
        plt.plot([], [], linewidth=2, color="#88CCEE", label="trend")
        
        ax.set_xlabel("$t$")
        ax.set_ylabel("$y$")
        ax.set_title("Mixture $\sum_{p = 1}^{p_\\mathrm{max}} "
                     "\\theta_p AR(p)$ model PPC")
        ax.legend(loc="upper right")

        if show:
            plt.show()
        return ax

    def get_model_probabilities(self):
        th = np.mean(self._fit_results.extract("theta")["theta"], axis=0)
        th = np.append(th[-1], th[:-1])
        return th

    def most_probable_model(self):
        th = self.get_model_probabilities()
        p_mp = np.argmax(th)  # Most probable order
        return p_mp

    def plot_poles(self, p=None, ax=None, show=False):
        if p is None:
            p_mp = self.most_probable_model()
        else:
            p_mp = p
        _fit_results = self._fit_results
        N_samples = len(_fit_results.extract("gamma")["gamma"])

        if p_mp == 0:
            # AR(0)
            roots = 0j + np.zeros(N_samples)
        else:
            ix0 = 1 + p_mp * (p_mp + 1) // 2 - p_mp
            b_params = [f"b[{tau}]"for tau in range(ix0, ix0 + p_mp)]
            param_df = _fit_results.to_dataframe(b_params)
            roots = _compute_roots(param_df.loc[:, b_params].to_numpy())

        # Z-plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        uc = patches.Circle((0, 0), radius=1, fill=False,
                            color='black', linestyle='dashed')
        ax.add_patch(uc)
        ax.scatter(roots.real, roots.imag, color="#882255",
                   marker="x", alpha=0.1)
        ax.set_title("System Poles of Model $p = {}$".format(p_mp))
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_aspect("equal")

        if show:
            plt.show()

        return ax

    def plot_posterior_params(self, ax=None, show=False):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        _fit_results = self._fit_results
        p_max = self.p_max

        th_params = [f"theta[{tau}]" for tau in [p_max + 1] +
                     list(range(1, p_max + 1))]
        th_params_tex = [f"$\\theta_{tau}$" for tau in range(p_max + 1)]

        gamma_params = ([f"gamma[{tau}]" for tau in range(1, p_max + 1)] +
                        [f"nu_gamma[{tau}]" for tau in range(1, p_max + 1)])
        gamma_params_tex = (
            [f"$\\Gamma_{tau}$" for tau in range(1, p_max + 1)] +
            [f"$\\mathrm{{log}}_{{10}}(\\nu_{tau})$"
             for tau in range(1, p_max + 1)])

        params = (["sigma", "nu_th", "mu", "r"] +
                  th_params + gamma_params)
        names = (["$\\sigma^2$", "$\\mathrm{log}_{10}(\\nu_\\theta)$",
                  "$\\mu_y$", "$r$"] + th_params_tex + gamma_params_tex)

        rename = {frm: to for frm, to in zip(params, names)}
        param_df = _fit_results.to_dataframe(params)

        for param in [p for p in gamma_params if p[:2] == "nu"]:
            param_df.loc[:, param] = np.log10(param_df.loc[:, param])
        param_df.loc[:, "nu_th"] = np.log10(param_df.loc[:, "nu_th"])
        param_df.loc[:, "sigma"] = param_df.loc[:, "sigma"]**2
        param_df = param_df.loc[:, params]\
            .rename(rename, axis=1)

        if ax is None:
            fig, ax = plt.subplots(1, 1)
            fig.suptitle("$y(t) \\sim \\mathcal{N}(\\mu_y + rt + "
                         "\\sum_{\\tau = 1}^p b_\\tau y(t - \\tau),"
                         " \\sigma^2); \\frac{1}{2}(1 + \\Gamma_\\tau) \\sim "
                         "\\beta_\\mu(\\mu_{\\tau, \\gamma}, \\nu_\\gamma); "
                         "p \\sim \mathrm{Cat}(\\theta)$")

        ax.set_title("Parameter Posteriors")
        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=300)
        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax)

        if show:
            plt.show()

        return ax

import os
from itertools import chain

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np
from numpy.polynomial.polynomial import polyroots

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler

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


class BayesAR(BaseEstimator, RegressorMixin, StanCacheMixin):
    def __init__(self, p=1, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True, max_samples_mem=500):
        BaseEstimator.__init__(self)
        StanCacheMixin.__init__(self, MODEL_DIR)

        self.p = p  # The model order
        self.stan_model, self.predict_model = self._load_compiled_models()

        self.stan_fitting_kwargs = {"chains": n_chains,
                                    "iter": samples_per_chain + warmup,
                                    "warmup": warmup, "init": "random",
                                    "init_r": 1.0, "n_jobs": n_jobs,
                                    "control": {"metric": "diag_e",
                                                "adapt_delta": 0.9}}

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
        data = {"T": T, "p": self.p, "y": X}

        fit_kwargs = self._setup_predict_kwargs(data, stan_fitting_kwargs)
        self._fit_results = self.stan_model.sampling(**fit_kwargs)

        def iter_tau(name):
            return [f"{name}[{tau}]" for tau in range(1, self.p + 1)]

        pars = ["mu", "r", "sigma", "nu_beta"] + list(chain.from_iterable(
            [iter_tau(name) for name in ["y0", "g_beta", "mu_beta", "g", "b"]]))

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

    def plot_ppc(self, y, ax=None, show=False):
        if ax is None:
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
        ax.set_title("AR(p) model PPC")
        ax.legend(loc="upper right")

        if show:
            plt.show()
        return ax

    def plot_posterior_params(self, ax=None, show=False):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        if ax is not None:
            raise NotImplementedError

        param_df = self._fit_results.to_dataframe()
        p = self.p

        b_params = [f"b[{tau}]"for tau in range(1, p + 1)]
        b_params_tex = [f"$b_{tau}$"for tau in range(1, p + 1)]

        roots = _compute_roots(param_df.loc[:, b_params].to_numpy())

        params = (["sigma", "nu_beta", "mu", "r"] + b_params)
        names = (["$\\sigma^2$", "$\\mathrm{log}_{10}(\\nu_\\beta)$",
                  "$\\mu_y$", "$r$"] + b_params_tex)
        rename = {frm: to for frm, to in zip(params, names)}

        param_df.loc[:, "nu_beta"] = np.log10(param_df.loc[:, "nu_beta"])
        param_df.loc[:, "sigma"] = param_df.loc[:, "sigma"]**2
        param_df = param_df.loc[:, params]\
            .rename(rename, axis=1)

        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()

        # Plot parameters
        fig.suptitle("$y(t) \\sim \\mathcal{N}(\\mu_y + rt + "
                     "\\sum_{\\tau = 1}^p b_\\tau y(t - \\tau), \\sigma^2); "
                     "\\frac{1}{2}(1 + \\Gamma_\\tau) \\sim "
                     "\\beta_\\mu(\\mu_\\beta, \\nu_\\beta)$")

        ax[0].set_title("Parameter Posteriors")
        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax[0])

        # Z-plot
        uc = patches.Circle((0, 0), radius=1, fill=False,
                            color='black', linestyle='dashed')
        ax[1].add_patch(uc)
        ax[1].scatter(roots.real, roots.imag, color="#882255",
                      marker="x", alpha=0.1)
        ax[1].set_title("System Poles")
        ax[1].set_xlabel("Re")
        ax[1].set_ylabel("Im")

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

        pars = ["mu", "r", "sigma", "nu_gamma", "gamma", "theta", "y0"]

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

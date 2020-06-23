import os

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyroots
from scipy.signal import freqz

from sklearn.base import RegressorMixin, BaseEstimator

from stanlearn.base import StanCacheMixin

try:
    MODEL_DIR = os.path.join(os.path.dirname(__file__),
                             "./stan_models/")
except NameError:  # no __file__ when interactive
    MODEL_DIR = "./stan_models/"


def _reorder_Bz(b):
    """
    Organizes b into B(z) = 1 + b[1]z^{-1} + ... + b[p]z^{-p}

    This is descending powers of z^{-1}
    """
    return np.hstack((np.ones(len(b)).reshape(-1, 1),
                      -np.atleast_2d(b)))


def _compute_roots(b):
    roots = []
    for bi in _reorder_Bz(b)[:, ::-1]:
        roots.append(polyroots(bi))
    return np.vstack(roots)


# This is basically an abstract base class -- I'm not using formal mechanisms
class BaseAR(StanCacheMixin):
    def __init__(self, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True):
        StanCacheMixin.__init__(self, MODEL_DIR)

        self.stan_model, self.predict_model = self._load_compiled_models()

        self.stan_fitting_kwargs = {"chains": n_chains,
                                    "iter_sampling": samples_per_chain,
                                    "iter_warmup": warmup, "inits": 0,
                                    "metric": "diag_e",
                                    "adapt_delta": 0.8}

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
        self._fit_results = self.stan_model.sample(**fit_kwargs)
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
                alpha=0.8, label="y\\_ppc")
        ax.plot([], [], linewidth=2, color="#88CCEE", label="trend")

        if labels:
            ax.set_xlabel("$t$")
            ax.set_ylabel("$y$")
            ax.set_title("AR(p) model PPC")
            ax.legend(loc="upper right")

        if show:
            plt.show()
        return ax

    def plot_posterior_basic(self, param_df, ax, scalar_params, vector_params,
                             modifier_map=None):
        """
        Plot AR(p) parameter posteriors.

        param_df should contain ["g", "sigma", "mu", "r"].

        'vector_params' should be a list of tuples (str, str_tex)
        where str is the name of the param in the model and str_tex is
        what it should be renamed to for nice plotting.  Each of these
        params should accept str.format(tau=tau).

        'scalar_params' similar, except does not need to be formatted.

        'modifier_map' maps from a param_name in scalar_params to
        a function which will be applied to the data before plotting, this
        is useful e.g. for taking logs or squaring.
        """
        p = self.p
        param_df = pd.DataFrame(param_df)  # Copy the input
        if modifier_map is None:
            modifier_map = []

        str_params = []
        tex_params = []
        for param in scalar_params:
            str_params.append(param[0])
            tex_params.append(param[1])
        for param in vector_params:
            str_params += [param[0].format(tau=tau) for tau in range(1, p + 1)]
            tex_params += [param[1].format(tau=tau) for tau in range(1, p + 1)]

        for param in modifier_map:
            param_df.loc[:, param] = param_df.loc[:, param]\
                .apply(modifier_map[param])

        rename = {frm: to for frm, to in zip(str_params, tex_params)}
        param_df = param_df.loc[:, str_params]\
            .rename(rename, axis=1)

        ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=300)
        ax.set_title(f"Parameter Posteriors")

        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax)
        return ax

    def _get_b(self, param_df, b_proto, tau_range):
        param_df = pd.DataFrame(param_df)  # Copy the input

        if tau_range is None:
            tau_range = range(1, self.p + 1)

        b_params = [b_proto.format(tau=tau) for tau in tau_range]

        b = param_df.loc[:, b_params].to_numpy()
        return b

    def plot_roots(self, param_df, ax, b_proto="b[{tau}]",
                   title="System Poles", tau_range=None):
        """
        Plot the poles of an AR(p) model.

        param_df should contain "b".
        """
        b = self._get_b(param_df, b_proto, tau_range)
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

    def plot_spectrum(self, param_df, ax=None, b_proto="b[{tau}]",
                      gain="sigma", tau_range=None, show=False,
                      title="$AR(p)$ Frequency Response"):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True)

        b = self._get_b(param_df, b_proto, tau_range)

        # TODO: Is this correct?  The gain just being sigma?
        K = param_df.loc[:, gain].to_numpy()
        num = K
        den = _reorder_Bz(b)

        w = np.linspace(0, np.pi, 512)
        h = []

        # TODO: Is my ordering of b correct?
        for i in range(len(K)):
            h.append(freqz(b=num[i], a=den[i], worN=w)[1])

        H = 20 * np.log10(np.abs(h))
        angle = np.unwrap(np.angle(h))

        ax[0].plot(w, H.T, linewidth=0.5, color="#CC6677", alpha=0.1)
        ax[0].plot(w, np.mean(H, axis=0),
                   linewidth=3.0, color="#882255", alpha=0.8)
        ax[1].plot(w, angle.T, linewidth=0.5, color="#CC6677", alpha=0.1)
        ax[1].plot(w, np.mean(angle, axis=0),
                   linewidth=3.0, color="#882255", alpha=0.8)

        ax[0].set_title("$20\\mathrm{log}_{10}|H(j\\omega)|$")
        ax[0].set_ylabel("Gain [dB]")

        ax[1].set_title("$\\angle H(j\\omega)$")
        ax[1].set_xlabel("Frequency [Rad / sample]")
        ax[1].set_ylabel("Phase [Rad]")

        fig.suptitle(title)

        if show:
            plt.show()

        return ax


class BayesAR(BaseAR, BaseEstimator, RegressorMixin):
    def __init__(self, p=1, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True):
        BaseEstimator.__init__(self)
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
            stan_fitting_kwargs: To be passed to cmdstanpy's .sampling method
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

        fig.suptitle("$\\Gamma$-parameterized AR(p) model with trend.")

        scalar_params = [("sigma", "$\\sigma^2$"),
                         ("mu", "$\\mu$"), ("r", "$r$")]
        vector_params = [("g[{tau}]", "$\Gamma_{{{tau}}}$")]
        modifier_map = {"sigma": lambda x: x**2}

        super().plot_posterior_basic(param_df, ax=ax[0],
                                     scalar_params=scalar_params,
                                     vector_params=vector_params,
                                     modifier_map=modifier_map)
        super().plot_roots(param_df, ax=ax[1], b_proto="b[{tau}]")

        if show:
            plt.show()
        return ax

    def plot_spectrum(self, ax=None, show=False):
        if ax is not None:
            raise NotImplementedError

        param_df = self._fit_results.to_dataframe(["b", "sigma"])
        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()

        super().plot_spectrum(param_df, b_proto="b[{tau}]", gain="sigma",
                              show=show)
        return ax


class BayesRepAR(BaseAR, BaseEstimator, RegressorMixin):
    def __init__(self, p=1, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True):
        BaseEstimator.__init__(self)
        BaseAR.__init__(self,
                        n_jobs=n_jobs, warmup=warmup,
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
            stan_fitting_kwargs: To be passed to cmdstanpy's .sampling method
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

        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        if k is not None:
            param_df = self._fit_results.to_dataframe(["b", "g", "sigma",
                                                       "mu", "r"])
            scalar_params = [(f"sigma[{k}]", "$\\sigma^2$"),
                             (f"mu[{k}]", "$\\mu$"), (f"r[{k}]", "$r$")]
            vector_params = [(f"g[{k},{{tau}}]", "$\\Gamma_{{{tau}}}$")]
            modifier_map = {f"sigma[{k}]": lambda x: x**2}

            super().plot_roots(param_df, ax=ax[1], b_proto=f"b[{k},{{tau}}]")
            fig.suptitle("$\\Gamma$-parameterized $\\mathrm{AR}(p)$ model "
                         f"with trend.  Sample $k = {k}$")

        else:
            param_df = self._fit_results.to_dataframe(
                ["nu_g", "b_hier", "g_hier", "sigma_hier", "nu_sigma",
                 "lambda"])
            scalar_params = [
                ("nu_g", "$\\mathrm{log}_{10}(\\nu_\\Gamma)$"),
                ("nu_sigma", "$\\mathrm{log}_{10}(\\bar{\\nu_\\sigma})$"),
                ("sigma_hier", "$\\bar{\\sigma}^2$"),
                ("lambda", "$\\lambda$")]
            vector_params = [("g_hier[{tau}]", "$\\bar{{\\Gamma}}_{tau}$")]
            modifier_map = {"sigma_hier": lambda x: x**2,
                            "nu_g": lambda x: np.log10(x),
                            "nu_sigma": lambda x: np.log10(x)}

            super().plot_roots(param_df, ax=ax[1], b_proto="b_hier[{tau}]",
                               title="Hierarchy Implied Poles")

        super().plot_posterior_basic(
            param_df, ax=ax[0], scalar_params=scalar_params,
            vector_params=vector_params, modifier_map=modifier_map)

        if show:
            plt.show()
        return ax

    def plot_spectrum(self, k=None, ax=None, show=False):
        if ax is not None:
            raise NotImplementedError

        p = self.p

        if k is not None:
            param_df = self._fit_results.to_dataframe(["b", "sigma"])
            b_proto = f"b[{k},{{tau}}]"
            gain = f"sigma[{k}]"
            title = f"$AR({p})$ Spectrum for $k = {k}$"
        else:
            param_df = self._fit_results.to_dataframe(["b_hier", "sigma_hier"])
            b_proto = "b_hier[{tau}]"
            gain = "sigma_hier"
            title = f"Hierarchical $AR({p})$ Spectrum"

        ax = super().plot_spectrum(param_df, b_proto=b_proto, gain=gain,
                                   title=title, ax=ax, show=show)
        return ax


class BayesMixtureAR(BaseAR, BaseEstimator, RegressorMixin):
    def __init__(self, p_max=1, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True, mu_th=None, nu_th=2):
        BaseEstimator.__init__(self)
        BaseAR.__init__(
            self, n_jobs=n_jobs, warmup=warmup,
            samples_per_chain=samples_per_chain,
            n_chains=n_chains, normalize=normalize)

        self.p = p_max  # The model order

        if mu_th is None:
            mu_th = np.ones(p_max + 1)
        mu_th = mu_th / sum(mu_th)  # Ensure it is a simplex

        if nu_th is None:
            nu_th = 1

        # In stan the last index is for the AR(0) model.
        self.mu_th = np.append(mu_th[1:], mu_th[0])
        self.nu_th = nu_th
        return

    def fit(self, X, y=None, sample_weight=None, **stan_fitting_kwargs):
        """
        "Fit" the model, that is, sample from the posterior.

        params:
            X (n_examples, m_features): Signal to fit, T x 1
            sample_weight: NotImplemented
            stan_fitting_kwargs: To be passed to cmdstanpy's .sampling method
        """
        if sample_weight is not None:
            raise NotImplementedError("sampling weighting is not implemented.")

        if len(X.shape) > 1:
            raise NotImplementedError
        T = len(X)

        data = {"T": T, "p_max": self.p, "y": X,
                "mu_th": self.mu_th, "nu_th": self.nu_th}
        pars = ["mu", "r", "sigma", "g", "b"]

        super().fit(data, stan_fitting_kwargs, pars)
        return

    def plot_ppc(self, y, ax=None, show=False, labels=False):
        y_ppc = self.get_ppc()
        y_trend = self.get_trend()
        ax = super().plot_ppc(y, y_ppc, y_trend, ax=ax,
                              show=show, labels=labels)
        return ax

    def get_model_probabilities(self):
        # The posterior of model probabilities
        th = np.mean(self._fit_results.extract("pz")["pz"], axis=0)
        th = np.append(th[-1], th[:-1])
        return th

    def most_probable_model(self):
        th = self.get_model_probabilities()
        p_mp = np.argmax(th)  # Most probable order
        return p_mp

    def plot_posterior_params(self, ax=None, show=False,
                              p=None):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        if ax is not None:
            raise NotImplementedError
        if p is None:
            p = self.most_probable_model()

        if p == 0:
            fig, ax = plt.subplots(1, 1)
            ax = [ax]
        else:
            fig, ax = plt.subplots(1, 2)

        param_df = self._fit_results.to_dataframe(["b", "g", "sigma",
                                                   "mu", "r", "pz"])

        fig.suptitle("Mixture of $\\Gamma$-parameterized AR(p) models")

        # Need to be able to do arithmetic...
        class FormatHack1:
            def format(self, tau=0):
                return "$p_{tau}$".format(tau=tau - 1)

        scalar_params = [("sigma", "$\\sigma^2$"),
                         ("mu", "$\\mu$"), ("r", "$r$")]
        vector_params = [("g[{tau}]", "$\\Gamma_{{{tau}}}$"),
                         ("pz[{tau}]", FormatHack1())]
        modifier_map = {"sigma": lambda x: x**2}

        super().plot_posterior_basic(
            param_df, ax=ax[0], scalar_params=scalar_params,
            vector_params=vector_params, modifier_map=modifier_map)
        if p > 0:
            ix0 = 1 + p * (p + 1) // 2 - p

            super().plot_roots(param_df, ax=ax[1], b_proto="b[{tau}]",
                               title=f"System Poles for $p = {p}$",
                               tau_range=range(ix0, ix0 + p))

        if show:
            plt.show()

        return ax

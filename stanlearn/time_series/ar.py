import os
from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np
from numpy.polynomial.polynomial import polyroots

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from stanlearn.base import StanCacheMixin

try:
    MODEL_DIR = os.path.join(os.path.dirname(__file__),
                             "./stan_models/")
except NameError:  # no __file__ when interactive
    MODEL_DIR = "./stan_models/"


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

    def _posterior(self, X, **stan_fitting_kwargs):
        N, M = X.shape

        if self.normalize:
            X = self._X_ss.transform(X)

        # y0, beta, sigma, nu = self._get_param_posterior()

        mem_samples = len(y0) * 8 * N / 1e6
        ss = int(1 + (mem_samples // self.max_samples_mem))  # Subsample
        K = len(y0[::ss])

        data = {"N": N, "M": M, "K": K}
        fit_kwargs = self._setup_predict_kwargs(data, stan_fitting_kwargs)
        fit_kwargs["iter"] = 1
        fit_kwargs["chains"] = 1

        predictions = self.predict_model.sampling(**fit_kwargs,
                                                  algorithm="Fixed_param")
        y_samples = predictions.extract("y")["y"][0, ...]
        y_hat = predictions.extract("y_hat")["y_hat"].ravel()

        if self.normalize:
            y_samples = np.vstack([self._y_ss.inverse_transform(y_s)
                                   for y_s in y_samples])
            y_hat = self._y_ss.inverse_transform(y_hat)
        return y_hat, y_samples

    def _get_aram_posterior(self):
        if self._fit_results is None:
            raise NotFittedError("Model isn't fit!")
        df = self._fit_results.to_dataframe()
        return

    def fit(self, X, y=None, sample_weight=None, **stan_fitting_kwargs):
        """
        "Fit" the model, that is, sample from the posterior.

        params:
            X (n_examples, m_features): Regressors
            y (n_examples): The targets
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
        # data = {"N": N, "M": M, "X": X, "y": y}
        data = {"T": T, "p": self.p, "y": X}

        fit_kwargs = self._setup_predict_kwargs(data, stan_fitting_kwargs)
        self._fit_results = self.stan_model.sampling(**fit_kwargs)
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
        y_ppc = self._fit_results.extract("y_ppc")["y_ppc"]
        return y_ppc

    def plot_posterior_params(self, show=False):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        param_df = self._fit_results.to_dataframe()
        p = self.p

        params = (["sigma", "lam", "mu"] +
                  [f"b[{tau}]"for tau in range(1, p + 1)])
        names = (["$\sigma$", "$\lambda$", "$\mu$"] + 
                  [f"$b_{tau}$" for tau in range(1, p + 1)])
        rename = {frm: to for frm, to in zip(params, names)}

        param_df = param_df.loc[:, params]\
            .rename(rename, axis=1)

        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()

        # Plot parameters
        fig.suptitle("$y(t) \sim \mathcal{N}(\mu + \sum_{\\tau = 1}^p "
                     "b_\\tau y(t - \\tau), \sigma^2); "
                     "b_\\tau \sim \mathcal{N}(0, \lambda^2)$")

        ax[0].set_title("Parameter Posteriors")
        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax[0])

        # Z-plot
        roots = self._compute_roots(param_df.iloc[:, 3:].to_numpy())

        uc = patches.Circle((0, 0), radius=1, fill=False,
                            color='black', linestyle='dashed')
        ax[1].add_patch(uc)
        ax[1].scatter(roots.real, roots.imag, color="#0072B2",
                      marker="x", alpha=0.2)
        ax[1].set_title("System Poles")

        if show:
            plt.show()
        return fig, ax

    def _compute_roots(self, b):
        roots = []
        for bi in b:
            roots.append(polyroots(np.append(-bi[::-1], 1)))
        return np.vstack(roots)

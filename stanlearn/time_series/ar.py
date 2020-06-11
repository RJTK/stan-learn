import os
from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True, max_samples_mem=500):
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
        self.max_samples_mem = max_samples_mem

        if normalize:
            self._y_ss = StandardScaler(with_mean=False)
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
        N, M = X.shape
        if M > 1:
            raise NotImplementedError

        if self.normalize:
            X = self._y_ss.fit_transform(X)

        X = X.ravel()
        # data = {"N": N, "M": M, "X": X, "y": y}
        data = {"N": N, "y": X}

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
        if ret_posterior:
            y_hat, y_samples = self._posterior(X, **stan_fitting_kwargs)
            return y_hat, y_samples
        else:
            return y_hat
    

    def plot_posterior_params(self, show=False):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        param_df = self._fit_results.to_dataframe()

        fig, ax = plt.subplots(1, 1)
        ax.set_title(
            "Parameter Posterior Marginals: "
            "$y \\sim \\mathcal{T}(\\nu, y_0 + X\\beta, \\sigma)$")
        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax)
        if show:
            plt.show()
        return fig, ax


if __name__ == "__main__":
    ar = BayesAR(normalize=False)
    T = 100
    v = 0.25 * np.random.normal(size=T)
    y = np.array(v)
    b = 0.6

    for t in range(1, T):
        y[t] = b * y[t - 1] + v[t]

    y = y[:, None]
    ar.fit(y)

    y_hat = ar._fit_results.extract("y_hat")["y_hat"]
    plt.plot(y_hat[::10].T, linewidth=0.5, color="m", alpha=0.25)
    plt.plot(y.ravel(), linewidth=2.0, color="b", alpha=0.8, label="y")
    plt.plot(np.mean(y_hat, axis=0), linewidth=2.0, color="r",
             alpha=0.8, label="y\_hat")
    plt.legend()
    plt.show()

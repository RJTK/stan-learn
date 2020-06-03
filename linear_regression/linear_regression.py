from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from stanlearn.base import StanCacheMixin


class BayesLinearRegression(BaseEstimator, RegressorMixin, StanCacheMixin):
    def __init__(self, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True, max_samples_mem=500):
        """
        An interface to the following stan model

        y0 ~ cauchy(0, 1);
        nu ~ cauchy(0, 1);
        sigma ~ normal(0, 1);  // half-normal
        lam ~ exponential(1);
        theta ~ normal(0, lam);
        y ~ student_t(nu, y0 + Q * theta, sigma);

        params:
          n_jobs: Number of cores to use
          warmup: Number of burnin iterations for HMC
          samples_per_chain: Number of samples to draw per chain
          n_chains: Number of chains (should run at least 2)
          normalize: Whether to normalize the data before feeding it
              to stan.  This is necessary as the priors in the model
              are fixed.
          max_samples_mem: A parameter to prevent blowing up all the
              memory when sampling the posterior predictive.
        """
        BaseEstimator.__init__(self)

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

    def _setup_predict_kwargs(self, data, extra_kwargs):
        fit_kwargs = deepcopy(self.stan_fitting_kwargs)
        fit_kwargs.update(extra_kwargs)
        fit_kwargs["data"] = data
        return fit_kwargs

    def _get_name(self):
        return type(self).__name__

    def _posterior(self, X, **stan_fitting_kwargs):
        N, M = X.shape

        if self.normalize:
            X = self._X_ss.transform(X)

        y0, beta, sigma, nu = self._get_param_posterior()

        # Ensure we don't use an excessive amount of memory
        # TODO: max_samples_mem is a massive underestimate of
        # TODO: the amount of memory used, why?
        mem_samples = len(y0) * 8 * N / 1e6
        ss = int(1 + (mem_samples // self.max_samples_mem))  # Subsample
        K = len(y0[::ss])

        data = {"N": N, "M": M, "K": K, "beta": beta[::ss], "y0": y0[::ss],
                "sigma": sigma[::ss], "X": X, "nu": nu[::ss]}
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

    def _get_param_posterior(self):
        if self._fit_results is None:
            raise NotFittedError("Model isn't fit!")
        df = self._fit_results.to_dataframe()
        M = sum(c[:4] == "beta" for c in df.columns)

        y0 = df.loc[:, "y0"].to_numpy()
        beta = df.loc[:, [f"beta[{j}]" for j in range(1, M + 1)]].to_numpy()
        sigma = df.loc[:, "sigma"].to_numpy()
        nu = df.loc[:, "nu"].to_numpy()

        return y0, beta, sigma, nu

    def fit(self, X, y, sample_weight=None, **stan_fitting_kwargs):
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

        if self.normalize:
            y = self._y_ss.fit_transform(y)
            X = self._X_ss.fit_transform(X)

        y = y.ravel()
        data = {"N": N, "M": M, "X": X, "y": y}

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
        y0, beta, _, _ = self._get_param_posterior()
        y0_mean = np.mean(y0)
        beta_mean = np.mean(beta, axis=0)
        if self.normalize:
            y_hat = y0_mean + self._X_ss.transform(X) @ beta_mean
            y_hat = self._y_ss.inverse_transform(y_hat)
        else:
            y_hat = y0_mean + X @ beta_mean

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
        M = sum([c[:4] == "beta" for c in param_df.columns])
        col_names = (["y0", "sigma", "nu"] +
                     [f"beta[{j}]" for j in range(1, M + 1)])
        var_names = (["$y_0$", "$\\sigma$", "$\\mathsf{log}_{10}(\\nu)$"] +
                     ["$\\beta_{{{}}}$".format(j)
                      for j in range(1, M + 1)])

        param_df.loc[:, "nu"] = np.log10(param_df.loc[:, "nu"])
        param_df = param_df.rename({frm: to for frm, to in zip(col_names,
                                                               var_names)},
                                   axis=1)

        param_df = param_df.loc[:, var_names]

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

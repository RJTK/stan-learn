import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from stanlearn.base import StanCacheMixin

MODEL_DIR = os.path.join(os.path.dirname(__file__),
                         "stan_models/")


class GaussianProcessRegression(BaseEstimator, RegressorMixin, StanCacheMixin):
    def __init__(self, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True, max_samples_mem=500):
        BaseEstimator.__init__(self)
        StanCacheMixin.__init__(self, MODEL_DIR)

        self.stan_model, self.predict_model = self._load_compiled_models()

        # The control parameters for NUTS, most are left as default
        control = {
            "metric": "diag_e",  # Type of mass matrix (diag_e default)
            "stepsize_jitter": 0.05,  # Slight randomization of stepsizes
            "adapt_engaged": True,
            "adapt_gamma": 0.05,  # Regularization scale
            "adapt_delta": 0.8,  # Target acceptance probability (.8 default)
            "adapt_kappa": 0.75,  # Relaxation exponent
            "adapt_t0": 10,  # Adaptation iteration offset
            "adapt_init_buffer": 75,  # First fast adapt period
            "adapt_term_buffer": 50,  # Last fast adapt period
            "adapt_window": 25,  # First slow adapt period
            "max_treedepth": 10,  # N_leapfrog ~ 2**max_treedepth
            }

        self.stan_fitting_kwargs = {"chains": n_chains,
                                    "iter": samples_per_chain + warmup,
                                    "warmup": warmup, "init": "random",
                                    "init_r": 1.0, "n_jobs": n_jobs,
                                    "control": control}

        self._fit_results = None
        self._fit_X = None
        self.normalize = normalize
        self.max_samples_mem = max_samples_mem

        if normalize:
            self._y_ss = StandardScaler(with_mean=True)
            self._X_ss = StandardScaler()
        return

    def _posterior(self, X, **stan_fitting_kwargs):
        N, M = X.shape
        Xt = self._fit_X
        Nt = Xt.shape[0]

        if self.normalize:
            X = self._X_ss.transform(X)

        y0, alpha, rho, nu, f, sigma = self._get_param_posterior()

        # Ensure we don't use an excessive amount of memory
        mem_samples = (len(y0) * 8 * N**2) / 1e6
        ss = int(1 + (mem_samples // self.max_samples_mem))  # Subsample
        K = len(y0[::ss])

        data = {"Nt": Nt, "N": N, "M": M, "K": K, "X": X, "Xt": Xt,
                "alpha": alpha[::ss], "rho": rho[::ss], "nu": nu[::ss],
                "sigma": sigma[::ss], "f": f[::ss], "y0": y0[::ss]}
        fit_kwargs = self._setup_predict_kwargs(data, stan_fitting_kwargs)
        fit_kwargs["iter"] = 1
        fit_kwargs["chains"] = 1

        predictions = self.predict_model.sampling(**fit_kwargs,
                                                  algorithm="Fixed_param")
        y_samples = predictions.extract("y_samples")["y_samples"][0, ...]
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

        y0 = df.loc[:, "y0"].to_numpy()
        alpha = df.loc[:, "alpha"].to_numpy()
        rho = df.loc[:, "rho"].to_numpy()
        nu = df.loc[:, "nu"].to_numpy()
        sigma = df.loc[:, "sigma"].to_numpy()

        f = df.loc[:, [c for c in df.columns if c[:2] == "f["]].to_numpy()
        return y0, alpha, rho, nu, f, sigma

    def fit(self, X, y, sample_weight=None, **stan_fitting_kwargs):
        if sample_weight is not None:
            raise NotImplementedError("sampling weighting is not implemented.")
        N, M = X.shape

        if self.normalize:
            y = self._y_ss.fit_transform(y)
            X = self._X_ss.fit_transform(X)

        y = y.ravel()
        data = {"N": N, "M": M, "X": X, "y": y}

        pars = ["y0", "alpha", "rho", "nu", "sigma", "f"]

        stan_fitting_kwargs.update({"pars": pars})
        fit_kwargs = self._setup_predict_kwargs(data, stan_fitting_kwargs)
        self._fit_results = self.stan_model.sampling(**fit_kwargs)
        self._fit_X = X

        print(self._fit_results.stansummary(
            pars=["y0", "alpha", "rho", "nu", "sigma"],
            probs=[0.1, 0.5, 0.9]))
        return

    def predict(self, X, ret_posterior=False, **stan_fitting_kwargs):
        y_hat, y_samples = self._posterior(X, **stan_fitting_kwargs)
        if ret_posterior:
            return y_hat, y_samples
        return y_hat

    def plot_posterior_params(self, show=False):
        """
        A helper method to plot the posterior parameter distribution.
        Will raise an error if .fit hasn't been called.
        """
        param_df = self._fit_results.to_dataframe()
        col_names = ["y0", "alpha", "rho", "nu", "sigma"]
        var_names = ["$y_0$", "$\\alpha$", "$\\rho$",
                     "$\\mathsf{log}_{10}(\\nu)$", "$\\sigma$"]

        param_df.loc[:, "nu"] = np.log10(param_df.loc[:, "nu"])
        param_df = param_df.loc[:, col_names]
        param_df = param_df.rename({frm: to for frm, to in zip(col_names,
                                                               var_names)},
                                   axis=1)

        fig, ax = plt.subplots(1, 1)
        ax.set_title(
            "Parameter Posterior Marginals: "
            "$y \\sim \\mathcal{T}(\\nu, y_0 + \mathcal{GP}(\\alpha, \\rho), "
            "\\sigma)$")
        sns.boxplot(
            data=param_df.melt(
                value_name="Posterior Samples",
                var_name="Parameter"),
            x="Parameter", y="Posterior Samples", ax=ax)
        if show:
            plt.show()
        return fig, ax

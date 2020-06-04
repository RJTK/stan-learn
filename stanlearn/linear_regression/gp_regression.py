from copy import deepcopy

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from stanlearn.base import StanCacheMixin


class GaussianProcess(BaseEstimator, RegressorMixin, StanCacheMixin):
    def __init__(self, n_jobs=-1, warmup=1000, samples_per_chain=1000,
                 n_chains=4, normalize=True, max_samples_mem=500):
        BaseEstimator.__init__(self)

        self.stan_model, self.predict_model = self._load_compiled_models()

        self.stan_fitting_kwargs = {"chains": n_chains,
                                    "iter": samples_per_chain + warmup,
                                    "warmup": warmup, "init": "random",
                                    "init_r": 1.0, "n_jobs": n_jobs,
                                    "control": {"metric": "diag_e",
                                                "adapt_delta": 0.9}}

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


if __name__ == "__main__":
    N = 10
    N_test = 5
    M = 2

    X_train = np.random.normal(size=(N, M))
    X_test = np.random.normal(size=(N, M))
    y_train = 22 + (0.5 * X_train[:, 0]**2 + + 0.3 * X_train[:, 1] + 
                    0.2 * np.random.normal(size=(N)))
    y_test = 22 + (0.5 * X_test[:, 0]**2 + 0.3 * X_test[:, 1]).ravel()

    # Test the noise robustness
    # y_train[12] = 66
    # y_train[19] = -91

    gp = GaussianProcess(warmup=1000, samples_per_chain=1000)
    gp.fit(X_train, y_train[:, None])
    # alpha, rho, sigma, y0 = gp._get_param_posterior()
    y_train_hat, y_train_post = gp.predict(X_train, ret_posterior=True)
    y_test_hat, y_test_post = gp.predict(X_test, ret_posterior=True)

    x = X_train[:, 0].ravel()
    asort = np.argsort(x)
    plt.plot(x[asort], y_train_post[:, asort].T, color="m", alpha=0.2,
             linewidth=0.5)
    plt.plot(x[asort], y_train[asort], label="train", color="b", marker="o")
    plt.plot(x[asort], y_train_hat[asort], label="hat", color="r", marker="o")
    plt.ylim(-100, 100)
    plt.legend()
    plt.show()

    x = X_test[:, 0].ravel()
    asort = np.argsort(x)
    plt.plot(x[asort], y_test_post[:, asort].T, color="m", alpha=0.2,
             linewidth=0.5)
    plt.plot(x[asort], y_test[asort], label="test", color="b", marker="o")
    plt.plot(x[asort], y_test_hat[asort], label="hat", color="r", marker="o")
    plt.ylim(-100, 100)
    plt.legend()
    plt.show()

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import r2_score
    from sklearn.decomposition import PCA

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train[:, None]
    gp = GaussianProcess(warmup=1000, samples_per_chain=1000)
    gp.fit(X_train, y_train)

    # fit_results = gp._fit_results

    gp = GaussianProcess(warmup=2000, samples_per_chain=1000,
                         normalize=True)
    gp._fit_results = fit_results
    gp._y_ss = StandardScaler().fit(y_train)
    gp._X_ss = StandardScaler().fit(X_train)
    gp._fit_X = X_train

    gp.plot_posterior_params(show=True)

    y_train_hat, y_train_post = gp.predict(X_train, ret_posterior=True)
    y_test_hat, y_test_post = gp.predict(X_test, ret_posterior=True)

    print("r2_train: {}".format(r2_score(y_train, y_train_hat)))
    print("r2_test: {}".format(r2_score(y_test, y_test_hat)))

    sns.regplot(y_train, y_train_hat)
    plt.title("PPC regplot")
    plt.show()

    sns.regplot(y_test, y_test_hat)
    plt.title("Test regplot")
    plt.show()

    # This is a terrible way to visualize
    pca = PCA(n_components=1).fit(X_train)
    x = pca.transform(X_train).ravel()
    asort = np.argsort(x)

    plt.plot(x[asort], y_train_post.T[asort], color="m", alpha=0.2,
             linewidth=0.5)
    plt.plot(x[asort], y_train[asort], label="train", color="b",
             linewidth=2)
    plt.plot(x[asort], y_train_hat[asort], label="hat", color="r",
             linewidth=2)
    plt.legend()
    plt.show()

    # # asort = np.argsort(X_train.ravel())
    # # plt.plot(X_train.ravel()[asort], y_train[asort])
    # # plt.show()

    # model = pystan.StanModel("./stan_models/GaussianProcess_model.stan",
    #                          include_paths="./stan_models/")
    # fit_model = model.sampling(chains=4, warmup=1000, data=data,
    #                            iter=2000)
    # df = fit_model.to_dataframe()
    # y_hat = df.loc[:, [f"y_test[{i}]" for i in range(1, N + 1)]].to_numpy()

    # pars = df.loc[:, ["alpha", "rho", "sigma"]]
    # pars = pars.melt()
    # sns.boxplot(data=pars, x="variable", y="value")
    # plt.show()

    # x = X_test.ravel()
    # x_train = X_train.ravel()
    # asort = np.argsort(x)
    # asort_train = np.argsort(x_train)

    # plt.plot(x[asort], y_hat[:, asort].T, linewidth=0.25, color="m", alpha=0.25)
    # plt.plot(x[asort], np.mean(y_hat, axis=0)[asort], linewidth=2, color="r",
    #          alpha=0.75, marker="o")
    # plt.plot(x[asort], y_test[asort], linewidth=2, color="b",
    #          alpha=0.75, marker="o")
    # plt.plot(x_train[asort_train], y_train[asort_train], linewidth=2, color="g",
    #          alpha=0.75, marker="o")
    # plt.show()

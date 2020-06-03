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
        self.normalize = normalize
        self.max_samples_mem = max_samples_mem

        if normalize:
            self._y_ss = StandardScaler(with_mean=False)
            self._X_ss = StandardScaler()
        return
    


if __name__ == "__main__":
    import pystan

    N = 100
    N_test = 100
    M = 1

    X_train = np.random.normal(size=(N, M))
    X_test = np.random.normal(size=(N, M))
    y_train = (0.5 * X_train**2 + 0.2 * np.random.normal(size=(N, M))).ravel()
    y_test = (0.5 * X_test**2).ravel()

    data = {"N": N, "N_test": N_test, "M": M, "X": X_train,
            "y": y_train, "X_test": X_test}

    asort = np.argsort(X_train.ravel())
    plt.plot(X_train.ravel()[asort], y_train[asort])
    plt.show()

    model = pystan.StanModel("./.models/GaussianProcess_model.stan")
    fit_model = model.sampling(chains=4, warmup=1000, data=data,
                               iter=2000)
    df = fit_model.to_dataframe()
    y_hat = df.loc[:, [f"y_test[{i}]" for i in range(1, N + 1)]].to_numpy()

    pars = df.loc[:, ["alpha", "rho", "sigma"]]
    pars = pars.melt()
    sns.boxplot(data=pars, x="variable", y="value")
    plt.show()

    x = X_test.ravel()
    x_train = X_train.ravel()
    asort = np.argsort(x)
    asort_train = np.argsort(x_train)

    plt.plot(x[asort], y_hat[:, asort].T, linewidth=0.25, color="m", alpha=0.25)
    plt.plot(x[asort], np.mean(y_hat, axis=0)[asort], linewidth=2, color="r",
             alpha=0.75, marker="o")
    plt.plot(x[asort], y_test[asort], linewidth=2, color="b",
             alpha=0.75, marker="o")
    plt.plot(x_train[asort_train], y_train[asort_train], linewidth=2, color="g",
             alpha=0.75, marker="o")
    plt.show()

import os

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import (load_diabetes, load_boston,
                              fetch_california_housing)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from stanlearn.linear_regression import BayesLinearRegression


FIGURE_DIR = os.path.join(os.path.dirname(__file__),
                          "./figures/")
try:
    os.mkdir(FIGURE_DIR)
except FileExistsError:
    pass


def do_example(X, y, name=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train[:, None]

    blr = BayesLinearRegression(normalize=True,
                                max_samples_mem=10)
    model = Pipeline(steps=[("qt", QuantileTransformer()),
                            ("blr", blr)])

    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train, ret_posterior=False)
    _, y_posterior_train =\
        model.predict(X_train, ret_posterior=True)
    y_hat_test = model.predict(X_test, ret_posterior=False)
    _, y_posterior_test = model.predict(X_test, ret_posterior=True)

    blr.plot_posterior_params(show=False)
    plt.savefig(FIGURE_DIR + f"{name}_params.png")
    plt.savefig(FIGURE_DIR + f"{name}_params.pdf")
    plt.show()

    asort = np.argsort(y_train.ravel())
    plt.plot(y_posterior_train.T[asort], linewidth=0.2, color="m", alpha=0.2)
    plt.plot([], [], color="m", label="Posterior Samples")
    plt.plot(y_train_hat[asort], linewidth=2, color="r", alpha=0.75,
             label="Posterior Mean")
    plt.plot(y_train[asort], linewidth=2, color="b", alpha=0.75,
             label="Training Data")
    plt.title("Prior Predictive Check")
    plt.xlabel("Sorted Examples")
    plt.ylabel("$y$")
    plt.legend(loc="upper left")
    plt.savefig(FIGURE_DIR + f"{name}_priorpc.png")
    plt.savefig(FIGURE_DIR + f"{name}_priorpc.pdf")
    plt.show()

    r2 = model.score(X_test, y_test)

    asort = np.argsort(y_test)
    plt.plot(y_posterior_test.T[asort], linewidth=0.5, color="m", alpha=0.2)
    plt.plot(y_hat_test[asort], linewidth=2, color="r", alpha=0.75)
    plt.plot(y_test[asort], linewidth=2, color="b", alpha=0.75)
    plt.plot([], [], color="m", label="Posterior Samples")
    plt.plot([], [], color="r", label="Posterior Mean")
    plt.plot([], [], color="b", label="True Values")
    plt.title("{} Prediction: $R^2 = {:0.3f}$".format(name, r2))
    plt.xlabel("Sorted Index")
    plt.ylabel("Target Value")
    plt.legend(loc="upper left")
    plt.savefig(FIGURE_DIR + f"{name}_pred.png")
    plt.savefig(FIGURE_DIR + f"{name}_pred.pdf")
    plt.show()

    sns.regplot(y_test, y_hat_test)
    plt.ylabel("Estimated Value")
    plt.xlabel("True Value")
    plt.title(f"{name} Estimates Regressiong Plot")
    plt.savefig(FIGURE_DIR + f"{name}_pred_regplot.png")
    plt.savefig(FIGURE_DIR + f"{name}_pred_regplot.pdf")
    plt.show()
    return


if __name__ == "__main__":
    do_example(*load_diabetes(return_X_y=True), name="Diabetes")
    do_example(*load_boston(return_X_y=True), name="Boston-Housing")
    do_example(*fetch_california_housing(return_X_y=True),
               name="California-Housing")

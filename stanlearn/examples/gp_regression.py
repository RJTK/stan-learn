import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

from stanlearn.linear_regression import GaussianProcessRegression


if __name__ == "__main__":
    N = 50
    N_test = 50
    M = 2

    X_train = np.random.normal(size=(N, M))
    X_test = np.random.normal(size=(N, M))
    y_train = 22 + (0.5 * X_train[:, 0]**2 + + 0.3 * X_train[:, 1] +
                    0.2 * np.random.normal(size=(N)))
    y_test = 22 + (0.5 * X_test[:, 0]**2 + 0.3 * X_test[:, 1]).ravel()

    # Test the noise robustness
    y_train[12] = 66
    y_train[19] = -91

    gp = GaussianProcessRegression(warmup=1000, samples_per_chain=1000)
    gp.fit(X_train, y_train[:, None])
    # alpha, rho, sigma, y0 = gp._get_param_posterior()
    y_train_hat, y_train_post = gp.predict(X_train, ret_posterior=True)
    y_test_hat, y_test_post = gp.predict(X_test, ret_posterior=True)

    gp.plot_posterior_params(show=True)

    x = X_train[:, 0].ravel()
    asort = np.argsort(x)
    plt.plot(x[asort], y_train_post[:, asort].T, color="m", alpha=0.2,
             linewidth=0.5)
    plt.plot(x[asort], y_train[asort], label="train", color="b", marker="o")
    plt.plot(x[asort], y_train_hat[asort], label="hat", color="r", marker="o")
    plt.ylim(-100, 100)
    plt.legend(loc="upper right")
    plt.show()

    x = X_test[:, 0].ravel()
    asort = np.argsort(x)
    plt.plot(x[asort], y_test_post[:, asort].T, color="m", alpha=0.2,
             linewidth=0.5)
    plt.plot(x[asort], y_test[asort], label="test", color="b", marker="o")
    plt.plot(x[asort], y_test_hat[asort], label="hat", color="r", marker="o")
    plt.ylim(-100, 100)
    plt.legend(loc="upper right")
    plt.show()

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train[:, None]
    gp = GaussianProcessRegression(warmup=1000, samples_per_chain=1000)
    gp.fit(X_train, y_train)

    fig, ax = gp.plot_posterior_params(show=False)
    fig.savefig("../../figures/gp_Diabetes_params.png")
    fig.savefig("../../figures/gp_Diabetes_params.pdf")
    plt.show()

    y_train_hat, y_train_post = gp.predict(X_train, ret_posterior=True)
    y_test_hat, y_test_post = gp.predict(X_test, ret_posterior=True)

    r2_train = r2_score(y_train, y_train_hat)
    r2_test = r2_score(y_test, y_test_hat)

    print("r2_train: {}".format(r2_train))
    print("r2_test: {}".format(r2_test))

    sns.regplot(y_train, y_train_hat)
    plt.title("PPC regplot")
    plt.show()

    sns.regplot(y_test, y_test_hat)
    plt.title("Test regplot")
    plt.show()

    asort = np.argsort(y_test)
    plt.plot(y_test_post.T[asort], color="m", alpha=0.5,
             linewidth=0.5)
    plt.plot([], [], color="m", label="posterior samples")
    plt.plot(y_test[asort], label="test data", color="b",
             linewidth=2)
    plt.plot(y_test_hat[asort], label="posterior mean", color="r",
             linewidth=2)
    plt.legend(loc="upper left")
    plt.title("(Sorted) Test Predictions $(R^2 = {:0.3f})$".format(r2_test))
    plt.xlabel("Sorted Index")
    plt.ylabel("$y$ Value")
    plt.savefig("../../figures/gp_Diabetes_pred.png")
    plt.savefig("../../figures/gp_Diabetes_pred.pdf")
    plt.show()

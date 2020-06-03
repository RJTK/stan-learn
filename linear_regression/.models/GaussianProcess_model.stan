/*
 * Simple Gaussian process model
 */

functions{
  matrix L_cov_exp_quad(vector[] X, real alpha, real rho){
    int N = size(X);
    matrix[N, N] K;

    K = cov_exp_quad(X, alpha, rho);

    for (i in 1:N)
      K[i, i] += 1e-9;

    return cholesky_decompose(K);
  }
}

data {
  int<lower=1> N;  // Number of training examples
  int<lower=1> N_test;  // Number of test points
  int<lower=1> M;  // Number of features

  vector[M] X[N];  // training regressors -- array of N M-vectors
  vector[N] y;  // target

  vector[M] X_test[N_test];  // testing regressors
}

transformed data {
  real delta = 1e-9;  // numerical necessity
}

parameters {
  real<lower=0> alpha;  // length scale
  real<lower=0> rho;  // gp scale
  real<lower=0> sigma;  // noise level
  real y0;  // mean offset
  vector[N] eta;  // noise to be transformed
}

model {
  vector[N] f;  // Latent GP
  matrix[N, N] L;  // Cholesky factor

  // Sample the GP parameters
  rho ~ inv_gamma(5, 5);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);

  y0 ~ cauchy(0, 1);

  L = L_cov_exp_quad(X, alpha, rho);  // Kernel

  // The "raw" gp noise
  eta ~ normal(0, 1);

  f = y0 + L * eta;  // The GP
  y ~ normal(f, sigma);  // The output
}

generated quantities{
  vector[N_test] y_test;

  vector[N_test] f_test;
  matrix[N_test, N_test] L_test;
  vector[N_test] eta_test;

  L_test = L_cov_exp_quad(X_test, alpha, rho);

  for (i in 1:N_test)
    eta_test[i] = normal_rng(0, 1);

  f_test = y0 + L_test * eta;
  for (i in 1:N_test)
    y_test[i] = normal_rng(f_test[i], sigma);
}

/*
 * Simple Gaussian process model
 */

#include stan_functions.stan

data {
  int<lower=1> N;  // Number of training examples
  int<lower=1> M;  // Number of features

  vector[M] X[N];  // training regressors -- array of N M-vectors
  vector[N] y;  // target
}

parameters {
  real<lower=0> alpha;  // length scale
  real<lower=0> rho;  // gp scale
  real<lower=0> sigma;  // noise level
  vector[N] eta;  // noise to be transformed
  real<lower=0> nu;  // T noise dof
  real y0;  // mean offset
}

transformed parameters {
  vector[N] f;  // Latent GP
  cholesky_factor_cov[N] L;  // Cholesky factor

  L = L_cov_exp_quad(X, alpha, rho);  // Kernel
  f = y0 + L * eta;  // The GP
}

model {
  eta ~ normal(0, 1);  // The "raw" gp noise

  // Sample the GP parameters
  rho ~ inv_gamma(3, 5);
  alpha ~ normal(0, 1);
  sigma ~ normal(0, 1);
  nu ~ cauchy(0, 1);

  // An offset
  // y0 ~ cauchy(0, 1);
  y0 ~ normal(0, 1);

  // The output
  // y ~ normal(f, sigma);
  y ~ student_t(nu, f, sigma);
}

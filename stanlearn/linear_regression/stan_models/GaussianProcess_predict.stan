#include stan_functions.stan

data {
  int<lower=1> Nt;  // Number of training points
  int<lower=1> N;  // Number of test points
  int<lower=1> M;  // Number of features
  int<lower=1> K;  // Number of posterior samples

  vector[M] Xt[Nt];  // Training regression matrix
  vector[M] X[N];  // Test regression matrix

  real<lower=0> alpha[K];  // length scale
  real<lower=0> rho[K];  // gp scale
  real<lower=0> nu[K];  // T noise dof
  vector[Nt] f[K];  // The latent GP from training
  real<lower=0> sigma[K];  // noise level
  real y0[K];  // mean offset
}

transformed data {
  vector[M] XXt[N + Nt];  // concatenated data

  XXt[:Nt] = Xt;
  XXt[Nt + 1:] = X;
}

generated quantities {
  vector[N] y_hat = rep_vector(0, N);  // The mean predictive
  vector[N] y_samples[K];  // Samples from the posterior predictive

  for (k in 1:K){
    real k_ = k;  // How do I typecast for arithmetic?

    // Compute the predicted mean of the latent GP at the test points
    // f*|f ~ N(K21 * K1^{-1} * f, K2 - K21 * K1^{-1} * K12)
    matrix[Nt + N, Nt + N] KXXt = K_cov_exp_quad(XXt, alpha[k], rho[k]);
    vector[N] f_hat = KXXt[Nt + 1:, :Nt] * mdivide_left_spd(KXXt[:Nt, :Nt], f[k]);

    // The matrix K2 - K21 * K1^{-1} * K12 (Schur complement) is just
    // The product L2 * L2' where L2 is the lower right Cholesky factor of K
    // TODO: Is there a way to get the chol factors of K + sI given only the
    // TODO: chol factors of K?
    matrix[N, N] LXt = cholesky_decompose(add_to_diag(KXXt, sigma[k]^2))[Nt + 1:, Nt + 1:];

    // Keep a running mean of y_hat to avoid storing extra stuff
    y_hat = ((k_ - 1) / k_) * y_hat + (1 / k_) * f_hat;

    // Draw samples from the predictive distribution
    // y_samples[k] = multi_normal_cholesky_rng(f_hat, LXt);
    y_samples[k] = multi_student_t_rng(nu[k], y_hat, LXt * LXt');
  }
}

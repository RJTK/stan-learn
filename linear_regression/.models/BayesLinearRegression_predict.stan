data{
  int<lower=0> N;  // Number of examples
  int<lower=0> M;  // Number of features
  int<lower=0> K;  // Number of posterior samples

  vector[K] y0;  // intercept term
  matrix[K, M] beta;  // model coefficients
  vector<lower=0>[K] sigma;  // noise term
  vector<lower=0>[K] nu;  // student-t dof

  matrix[N, M] X;  // regressors
}

transformed data{
  matrix[K, N] y_hat_samples;  // y0 + Xbeta

  y_hat_samples = rep_matrix(y0, N) + beta * X';
}

generated quantities{
  real y[K, N];  // samples from posterior predictive
  vector[N] y_hat;  // The posterior mean prediction

  for (k in 1:K)
    y[k, :] = student_t_rng(nu[k], y_hat_samples[k, :], sigma[k]);

  for (n in 1:N)
    y_hat[n] = mean(y_hat_samples[:, n]);
}

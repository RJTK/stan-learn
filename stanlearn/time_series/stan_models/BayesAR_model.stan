/*
 * A simple AR(p) model.
 */

#include ar_functions.stan

data {
  int<lower=1> T;  // Number of examples
  int<lower=1> p;  // The model order
  vector[T] y;  // the data series y(t)
}

transformed data {
  vector[T] t;  // "time"
  vector[p + 1] t0;  // Time before t=0
  for(i in 1:T)
    t[i] = i;
  for(i in 1:p + 1)
    t0[i] = -p + i;
  t = t / T;  // Normalize to between [1 / T, 1]
  t0 = t0 / T;  // Normalize to between [-(p - 1) / T, 1 / T]
}

parameters {
  real mu;  // Mean value
  real r;  // Linear trend coefficient
  vector[p] y0;
  vector<lower=0, upper=1>[p] g_beta;  // For the reflection coefficients
  real<lower=0> sigma;  // noise level
}

transformed parameters {
  vector[p] b;  // AR Coefficients
  vector<lower=-1, upper=1>[p] g;  // Reflection coefficients

  g = 2 * g_beta - 1;  // transform to (-1, 1)
  b = step_up(g);  // Compute the actual AR coefficients
}

model {
  vector[p] alpha;  // Params for beta prior on g
  vector[p] beta;
  vector[T] trend;
  vector[p + 1] trend0;
  matrix[p + 1, p + 1] L;

  // Noise level in the signal (i.e. eps_p)
  sigma ~ inv_gamma(1, 1);

  // Uniform prior on reflection coefficients
  g ~ uniform(-1, 1);

  // trend parameters
  r ~ normal(0, 1);  // The linear time coefficient
  mu ~ normal(0, 1);  // A mean offset

  trend = mu + r * t;
  trend0 = mu + r * t0;

  // Sample y0 s.t. we have stationarity
  // L is the Chol-factor of symtoep(r), r = [r(0)..r(p)] the autocorr seq
  // L = chol_factor_g(g, sigma);
  // target += multi_normal_cholesky_lpdf(append_row(y0, y[1]) | trend0, L);

  y0 ~ normal(0, 1);

  // The actual AR model
  y - trend ~ ar_model(y0, b, sigma);
}

generated quantities {
  vector[T] y_ppc;
  real y_ll;
  vector[T] trend;
  vector[p] trend0;

  trend = mu + r * t;

  y_ll = ar_model_lpdf(y - trend | y0, b, sigma);
  y_ppc = trend + ar_model_rng(y, y0, b, sigma);
}

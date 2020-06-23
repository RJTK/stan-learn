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
  vector[p] t0;  // Time before t=0
  for(i in 1:T)
    t[i] = i;
  for(i in 1:p)
    t0[i] = i;

  t = t / T;  // Normalize to between [-(p - 1) / T, 1]
  t0 = (t0 - p) / T;  // Normalize to between [-(p - 1) / T, 1]
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
  vector[p] trend0;
  vector[p + 1] r_full;
  matrix[p + 1, p + 1] L;

  // Noise level in the signal (i.e. eps_p)
  sigma ~ inv_gamma(1, 1);

  // Uniform prior on reflection coefficients
  g ~ uniform(-1, 1);

  // trend parameters
  r ~ normal(0, 1);  // The linear time coefficient
  mu ~ normal(0, 1);  // A mean offset

  trend = mu + r * t;
  trend0 = mu + t0;

  // Sample y0 s.t. we have stationarity
  y0 - trend0 ~ ar_initial_values(y[1] - trend[1], g, sigma);

  // The actual AR model
  y - trend ~ ar_model(y0, b, sigma);
}

generated quantities {
  vector[T] y_ppc;
  real y_ll;
  vector[T] trend;

  trend = mu + r * t;

  y_ll = ar_model_lpdf(y - trend | y0, b, sigma);
  y_ppc = trend + ar_model_rng(y, y0, b, sigma);
}

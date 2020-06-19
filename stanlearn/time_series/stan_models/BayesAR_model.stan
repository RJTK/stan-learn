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
    t0[i] = -p + i;
  t = t / T;  // Normalize to between [1 / T, 1]
  t0 = t0 / T;  // Normalize to between [-(p - 1) / T, 0]
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
  vector[T] trend = mu + r * t;
  vector[p] trend0 = mu + r * t0;

  g = 2 * g_beta - 1;  // transform to (-1, 1)
  b = step_up(g);  // Compute the actual AR coefficients
}

model {
  vector[p] alpha;  // Params for beta prior on g
  vector[p] beta;

  // Noise level in the signal
  sigma ~ inv_gamma(1, 1);

  // Prior for the reflection coefficients
  g_beta ~ uniform(0, 1);

  // trend parameters
  r ~ normal(0, 1);  // The linear time coefficient
  mu ~ normal(0, 1);  // A mean offset

  // Should sample from the stationary distribution
  y0 ~ normal(trend0, sigma);

  // The actual AR model
  y - trend ~ ar_model(y0, b, sigma);
}

generated quantities {
  vector[T] y_ppc;
  real y_ll;
  y_ll = ar_model_lpdf(y - trend | y0, b, sigma);
  y_ppc = trend + ar_model_rng(y, y0, b, sigma);
}

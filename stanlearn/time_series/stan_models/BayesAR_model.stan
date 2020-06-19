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
  real<lower=0> sigma_hier;  // mean param hierarchy on noise level
  real<lower=0> sigma_rate;
  real<lower=0> sigma;  // noise level

  vector<lower=0, upper=1>[p] mu_beta;  // Mean vector for g_beta
  real<lower=0> nu_beta;  // pseudo-samples on g_beta
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
  sigma_hier ~ normal(0, 1);
  sigma_rate ~ exponential(1);
  sigma ~ gamma(sigma_hier * sigma_rate, sigma_rate);

  // Priors for the reflection coefficients
  mu_beta ~ uniform(0, 1);  // A p-vector
  nu_beta ~ inv_gamma(3, 3);  // keep (alpha, beta) > 1 else we get a U shape
  alpha = mu_beta * nu_beta;
  beta = (1 - mu_beta) * nu_beta;
  g_beta ~ beta(alpha, beta);  // in (0, 1)

  // trend parameters
  r ~ normal(0, 2);  // The linear time coefficient
  mu ~ normal(0, 2);  // A mean offset

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

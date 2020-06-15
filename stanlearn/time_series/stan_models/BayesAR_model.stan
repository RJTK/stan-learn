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
  vector[p + T] t;  // "time"
  for (i in -p + 1:T)
    t[i + p] = i;
  t = t / T;  // Normalize to between [-p/T, 1]
}

parameters {
  real mu;  // Mean value
  real r;  // Linear trend coefficient
  vector[p] y0;  // Initial values
  vector<lower=0, upper=1>[p] g_beta;  // For the reflection coefficients
  real<lower=0> sigma;  // noise level

  vector<lower=0, upper=1>[p] mu_beta;  // Mean vector for g_beta
  real<lower=0> nu_beta;  // pseudo-samples on g_beta
}

transformed parameters {
  vector[p] b;  // AR Coefficients
  vector<lower=-1, upper=1>[p] g;  // Reflection coefficients
  vector[p + T] trend = mu + r * t;

  g = 2 * g_beta - 1;  // transform to (-1, 1)

  b = step_up(g);  // Compute the actual AR coefficients
}

model {
  vector[p] alpha;  // Params for beta prior on g
  vector[p] beta;

  // Noise level in the signal
  sigma ~ normal(0, 5);

  // Priors for the reflection coefficients
  mu_beta ~ uniform(0, 1);  // A p-vector
  nu_beta ~ inv_gamma(3, 3);  // Want to keep (alpha, beta) > 1 else we get a U shape
  alpha = mu_beta * nu_beta;
  beta = (1 - mu_beta) * nu_beta;
  g_beta ~ beta(alpha, beta);  // in (0, 1)

  // trend parameters
  r ~ normal(0, 2);  // The linear time coefficient
  mu ~ normal(0, 2);  // A mean offset

  y0 ~ normal(trend[:p], sigma);  // Initial values
  y - trend[p + 1:] ~ ar_model(y0, b, sigma);
}

generated quantities {
  vector[T] y_ppc;
  y_ppc = trend[p + 1:] + ar_model_rng(y, y0, b, sigma);
}

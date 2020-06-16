/*
 * A simple AR(p) model.
 */

#include ar_functions.stan

data {
  int<lower=1> Tfull;  // Number of examples
  int<lower=1> K;  // Number of realizations
  int<lower=1> p;  // The model order
  vector[Tfull] yfull[K];  // the data series y(t)
  // vector[Tfull] yfull;  // the data series y(t)
}

transformed data {
  int<lower=1> T = Tfull - p;
  vector[T] y[K];
  vector[p] y0[K];
  vector[T] t;  // "time"
  for (i in 1:T)
    t[i] = i;
  t = t / T;  // Normalize to between [0, 1]

  y0 = yfull[:, :p];
  y = yfull[:, p + 1:];
}

parameters {
  real mu;  // Mean value
  real r;  // Linear trend coefficient
  vector<lower=0, upper=1>[p] g_beta;  // For the reflection coefficients
  real<lower=0> sigma_hier;  // mean param hierarchy on noise level
  real<lower=0> sigma_rate;
  real<lower=0> sigma[K];  // noise level

  vector<lower=0, upper=1>[p] mu_beta;  // Mean vector for g_beta
  real<lower=0> nu_beta;  // pseudo-samples on g_beta
}

transformed parameters {
  vector[p] b;  // AR Coefficients
  vector<lower=-1, upper=1>[p] g;  // Reflection coefficients
  vector[T] trend = mu + r * t;

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

  for(k in 1:K)
    y[k] - trend ~ ar_model(y0[k], b, sigma[k]);
}

generated quantities {
  vector[T] y_ppc[K];
  for(k in 1:K)
    y_ppc[k] = trend + ar_model_rng(y[k], y0[k], b, sigma[k]);
}

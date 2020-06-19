/*
 * AR(p) models allowing for multiple repetitions (i.e. arrays
 * of AR models).  We fit a hierarchy over basic AR models.
 */

#include ar_functions.stan

data {
  int<lower=1> T;  // Number of examples
  int<lower=1> K;  // Number of realizations
  int<lower=1> p;  // The model order
  vector[T] y[K];  // the data series y(t)
}

transformed data {
  vector[T] t;  // "time"
  vector[p] t0;  // Initial time
  for(i in 1:T)
    t[i] = i;
  for(i in 1:p)
    t0[i] = i - p;
  t = t / T;  // Normalize to between [1/T, 1]
  t0 = t0 / T;  // Normalize to [-(p - 1)/T, 0]
}

parameters {
  real mu[K];  // Mean value
  real r[K];  // Linear trend coefficient
  vector[p] y0[K];  // Initial values
  vector<lower=0, upper=1>[p] mu_beta;  // Mean vector for g_beta
  vector<lower=0, upper=1>[p] g_beta[K];  // reflection coefs
  real<lower=0> sigma_hier;  // mean param hierarchy on noise level
  real<lower=0> nu_sigma;
  real<lower=0> sigma[K];  // noise level

  real<lower=0> nu_g;  // pseudo-samples on g_beta
}

transformed parameters {
  vector[p] b[K];  // AR Coefficients
  vector<lower=-1, upper=1>[p] g[K];  // Reflection coefficients
  vector[T] trend[K];
  vector[p] trend0[K];

  for(k in 1:K){
    trend[k] = mu[k] + r[k] * t;
    trend0[k] = mu[k] + r[k] * t0;
    g[k] = 2 * g_beta[k] - 1;  // transform to (-1, 1)
    b[k] = step_up(g[k]);  // Compute the actual AR coefficients
  }
}

model {
  vector[p] alpha;  // Params for beta prior on g
  vector[p] beta;

  for(k in 1:K)  // TODO: sample from stationary
    y0[k] ~ normal(trend0[k], sigma[k]);

  // Noise level in the signal
  sigma_hier ~ normal(0, 1);
  nu_sigma ~ exponential(1);
  sigma ~ gamma(sigma_hier * nu_sigma, nu_sigma);

  // Priors for the reflection coefficients
  mu_beta ~ uniform(0, 1);  // A p-vector
  nu_g ~ inv_gamma(3, 3);  // keep (alpha, beta) > 1 else we get a U shape
  alpha = mu_beta * nu_g;
  beta = (1 - mu_beta) * nu_g;
  for(k in 1:K)
    g_beta[k] ~ beta(alpha, beta);  // in (0, 1)

  // trend parameters -- no hierarchy on these
  r ~ normal(0, 2);  // The linear time coefficient
  mu ~ normal(0, 2);  // A mean offset

  for(k in 1:K)
    y[k] - trend[k] ~ ar_model(y0[k], b[k], sigma[k]);
}

generated quantities {
  vector<lower=-1, upper=1>[p] g_hier;  // Hierarchical refl coefs
  vector[p] b_hier;  // Hierarchical implied AR coefs
  vector[T] y_ppc[K];
  real y_ll[K];

  g_hier = 2 * mu_beta - 1;
  b_hier = step_up(g_hier);  // Compute the actual AR coefficients

  for(k in 1:K){
    y_ll[k] = ar_model_lpdf(y[k] - trend[k] | y0[k], b[k], sigma[k]);
    y_ppc[k] = trend[k] + ar_model_rng(y[k], y0[k], b[k], sigma[k]);
  }
}

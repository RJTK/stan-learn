/*
 * A mixture of AR(p) models up to a maximum order p_max.
 * Information on ragged datastructures in stan is relevant
 */

#include ar_functions.stan

data {
  int<lower=1> T;  // Number of examples
  int<lower=1> p_max;  // The model order

  simplex[p_max + 1] mu_th;  // Dirichlet mean for theta

  vector[T] y;  // the data series y(t)
}

transformed data {
  vector[p_max + T] t;  // "time"
  int n_params = (p_max * (p_max + 1)) / 2;
  int model_size[p_max];

  for(p in 1:p_max)  // For a raged data structure
    model_size[p] = p;

  for(i in -p_max + 1:T)
    t[i + p_max] = i;
  t = t / T;  // Normalize to between [-p/T, 1]
}

parameters {
  real mu;  // Mean value
  real r;  // Linear trend coefficient
  vector[p_max] y0;  // Initial values
  simplex[p_max + 1] theta;  // Mixture parameters

  vector<lower=0, upper=1>[p_max] mu_gamma;  // Mean vector for gamma
  real<lower=0> nu_th; // pseudo-samples for Dirichlet
  vector<lower=0>[p_max] nu_gamma;  // pseudo-samples gamma
  vector<lower=-1, upper=1>[p_max] gamma;  // Reflection coefficients

  real<lower=0> sigma;  // noise level
}

transformed parameters {
  vector[n_params] b;  // AR Coefficients

  {int pos = 1;
    // Compute the actual AR coefficients
    for(p in 1:p_max){
      b[pos:pos + model_size[p] - 1] = step_up(gamma[:p]);
      pos += model_size[p];}}}

model {
  vector[p_max] alpha;  // transformed gamma prior params
  vector[p_max] beta;
  vector[p_max + 1] alpha_th;  // transformed Dirichlet prior for theta

  vector[p_max + 1] lpdfs;  // mixture pdfs
  vector[p_max + T] trend;  // trend term

  // real mu_gamma = 0.5;
  mu_gamma ~ uniform(0, 1);

  nu_th ~ inv_gamma(3, 3);
  alpha_th = mu_th * nu_th;

  theta ~ dirichlet(alpha_th);

  // Noise level in the signal
  sigma ~ normal(0, 5);

  // Priors for the reflection coefficients
  // mu_gamma ~ uniform(0, 1);  // A vector
  // nu_gamma ~ inv_gamma(3, 3);
  nu_gamma ~ exponential(1);
  alpha = mu_gamma .* nu_gamma;
  beta = (1 - mu_gamma) .* nu_gamma;

  // trend parameters
  r ~ normal(0, 2);  // The linear time coefficient
  mu ~ normal(0, 2);  // A mean offset

  // The simple trend term
  trend = mu + r * t;

  // Initial values
  y0 ~ normal(trend[:p_max], sigma);

  // Mixture AR(p), including "AR(0)" (pure noise)
  {int pos = 1;
    vector[T] detrend_y = y - trend[p_max + 1:];
    for(p in 1:p_max){
      lpdfs[p] = theta[p] + ar_model_lpdf(detrend_y | y0[:model_size[p]],
                                          b[pos:pos + model_size[p] - 1],
                                          sigma);
      pos += model_size[p];
    }
    lpdfs[p_max + 1] = theta[p_max + 1] + normal_lpdf(detrend_y | 0, sigma);
  }

  target += beta_lpdf(0.5 * (1 + gamma) | alpha, beta);  // in (0, 1)
  target += log_sum_exp(lpdfs);
}

generated quantities {
  vector[T] y_ppc;
  vector[T] trend;  // trend term
  int p = categorical_rng(theta);
  int pos;

  trend = mu + r * t[p_max + 1:];
  y_ppc = trend;

  if(p == 1){  // AR(1)
    y_ppc += ar_model_rng(y, y0[:1], b[:1], sigma);
  }
  else if(p == p_max + 1){  // AR(0)
    for(i in 1:T)
      y_ppc[i] += normal_rng(0, sigma);
  }
  else{  // AR(p), p > 1
    pos = 1 + sum(model_size[:p - 1]);
    y_ppc += ar_model_rng(y, y0[:model_size[p]], b[pos:pos + model_size[p] - 1], sigma);
  }
}

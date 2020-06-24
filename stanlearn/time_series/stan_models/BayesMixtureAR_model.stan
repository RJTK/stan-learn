/*
 * A mixture of AR(p) models up to a maximum order p_max.
 * Information on ragged datastructures in stan is relevant
 */

#include ar_functions.stan

data {
  int<lower=1> T;  // Number of examples
  int<lower=1> p_max;  // The model order

  simplex[p_max + 1] mu_th;  // Dirichlet mean for theta
  real<lower=0> nu_th;  // Pseudo-samples for Dirichlet

  vector[T] y;  // the data series y(t)
}

transformed data {
  vector[T] t;  // "time"
  vector[p_max] t0;  // Initial time before t=0
  int n_params = (p_max * (p_max + 1)) / 2;
  int model_size[p_max];

  for(p in 1:p_max)  // For a raged data structure
    model_size[p] = p;

  for(i in 1:T)
    t[i] = i;
  for(i in 1:p_max)
    t0[i] = i;
  t = t / T;  // Normalize to between [1/T, 1]
  t0 = (t0 - p_max) / T;  // [-(p_max - 1) / T, 0]
}

parameters {
  simplex[p_max + 1] theta;  // Mixture parameters

  // Trend params
  real mu;  // Mean value
  real r;  // Linear trend coefficient

  // AR(p) params
  vector[p_max] y0;  // Initial values
  vector<lower=-1, upper=1>[p_max] g;  // Reflection coefficients

  real<lower=0> sigma;  // noise level
}

transformed parameters {
  vector[n_params] b;  // AR Coefficients
  vector[p_max + 1] lpdfs;  // mixture pdfs
  vector[T] trend;  // trend term
  vector[p_max] trend0;  // trend term

  // The simple trend term
  trend = mu + r * t;
  trend0 = mu + r * t0;

  {int pos = 1;
    // Compute the actual AR coefficients
    for(p in 1:p_max){
      b[pos:pos + model_size[p] - 1] = step_up(g[:p]);
      pos += model_size[p];}}

  {int pos = 1;
    vector[T] detrend_y = y - trend;
    for(p in 1:p_max){
      lpdfs[p] = theta[p] + ar_model_lpdf(detrend_y | y0[:model_size[p]],
                                          b[pos:pos + model_size[p] - 1],
                                          sigma);
      pos += model_size[p];
    }
    lpdfs[p_max + 1] = theta[p_max + 1] + normal_lpdf(detrend_y | 0, sigma);
  }
}

model {
  vector[p_max + 1] alpha_th;  // transformed Dirichlet prior for theta

  g ~ uniform(-1, 1);  // Uniform refl coefs
  theta ~ dirichlet(mu_th * nu_th);  // Uniform

  // Noise level in the signal
  sigma ~ inv_gamma(1, 1);

  // trend parameters
  r ~ normal(0, 1);  // The linear time coefficient
  mu ~ normal(0, 1);  // A mean offset

  // Initial values
  y0 - trend0 ~ ar_initial_values(y[1] - trend[1], g, sigma);

  // Mixture AR(p), including "AR(0)" (pure noise)
  target += log_sum_exp(lpdfs);
}

generated quantities {
  vector[T] y_ppc;
  real y_ll;
  vector[p_max + 1] pz = softmax(lpdfs);  // Model probabilites

  y_ppc = trend;

  { int pos;
    int p = categorical_rng(pz);  // posterior mixture distr.
    y_ll = lpdfs[p];

    if(p == 1){  // AR(1)
      y_ppc += ar_model_rng(y, y0[:1], b[:1], sigma);
      y_ll += ar_model_lpdf(y - trend | y0[:1], b[:1], sigma);}
    else if(p == p_max + 1){  // AR(0)
      for(i in 1:T){
        y_ppc[i] += normal_rng(0, sigma);
        y_ll += normal_lpdf(y - trend | 0, sigma);}}
    else{  // AR(p), p > 1
      pos = 1 + sum(model_size[:p - 1]);

      y_ppc += ar_model_rng(y, y0[:model_size[p]],
                            b[pos:pos + model_size[p] - 1], sigma);
      y_ll += ar_model_lpdf(y - trend | y0[:model_size[p]],
                            b[pos:pos + model_size[p] - 1], sigma);
    }}}

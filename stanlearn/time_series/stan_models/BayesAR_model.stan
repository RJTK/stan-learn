/*
 * A simple AR(p) model.
 */

functions {
  vector step_up(vector g){
    /*
     * Maps a vector of p reflection coefficients g (|g| < 1) to a
     * stable sequence b of AR coefficients.
     */
    int p = dims(g)[1];  // Model order
    vector[p] b;  // AR Coefficients
    vector[p] b_cpy;  // Memory

    b[1] = g[1];
    for(k in 1:p - 1){
        b_cpy = b;
        for(tau in 1:k){
          b[tau] = b_cpy[tau] + g[k + 1] * b_cpy[k - tau + 1];
        }
        b[k + 1] = g[k + 1];
      }

    return -b;
  }
}

data {
  int<lower=1> T;  // Number of examples
  int<lower=1> p;  // The lag
  vector[T] y;  // the data series y(t)
}

transformed data {
}

parameters {
  real mu;  // Mean value
  real y0[p];  // Initial values
  vector<lower=0, upper=1>[p] g_beta;  // For the reflection coefficients
  real<lower=0> sigma;  // noise level

  vector<lower=0, upper=1>[p] mu_beta;  // Mean vector for g_beta
  real<lower=0> nu_beta;  // pseudo-samples on g_beta
}

transformed parameters {
  vector[p] b;  // AR Coefficients
  vector<lower=0>[p] alpha;  // Params for beta prior on g
  vector<lower=0>[p] beta;
  vector<lower=-1, upper=1>[p] g;  // Reflection coefficients

  g = 2 * g_beta - 1;  // transform to (-1, 1)
  alpha = mu_beta * nu_beta;
  beta = (1 - mu_beta) * nu_beta;

  b = step_up(g);  // Compute the actual AR coefficients
}

model {
  // Noise level in the signal
  sigma ~ normal(0, 5);

  // Priors for the reflection coefficients
  mu_beta ~ uniform(0, 1);  // A p-vector
  nu_beta ~ student_t(3, 1, 5);  // A scalar
  g_beta ~ beta(alpha, beta);  // in (0, 1)

  mu ~ normal(0, 1);  // A mean offset
  y0 ~ normal(0, 1);  // The initial values

  for(t in 1:p)
    y[t] ~ normal(mu + y0[t], sigma^2);

  for(t in p + 1:T){
    real y_hat = 0;
    for(tau in 1:p)
      y_hat += b[tau] * y[t - tau];
    y[t] ~ normal(mu + y_hat, sigma^2);
  }
}

generated quantities {
  vector[T] y_ppc;

  for(t in 1:p)
    y_ppc[t] = normal_rng(mu + y0[t], sigma^2);

  for(t in p + 1:T){
    real y_hat = 0;
    for(tau in 1:p)
      y_hat += b[tau] * y[t - tau];
    y_ppc[t] = normal_rng(mu + y_hat, sigma^2);
  }
}

/*
 * A simple AR(p) model.
 */

functions {
  vector step_up_recursion(vector g){
    /*
     * Maps a vector of p reflection coefficients g (|g| < 1) to a
     * stable sequence b of AR coefficients.
     */
    int p = dims(g)[1];  // Model order
    vector[p + 1] b;  // AR Coefficients
    vector[p + 1] b_cpy;  // Memory

    b[1] = 1;
    for(k in 1:p){
        b_cpy = b;
        for(tau in 2:k){
          b[tau] = b_cpy[tau] + g[k] * b_cpy[k - tau + 1];
        }
        b[k + 1] = g[k];
      }

    return -b[2:];
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
  vector<lower=-1, upper=1>[p] g;  // Reflection coefficients
  real<lower=0> sigma;  // noise level
  real<lower=0> lam;  // variance of reflection coefficients
}

transformed parameters {
  vector[p] b;  // AR Coefficients

  b = step_up_recursion(g);
}

model {
  sigma ~ normal(0, 1);
  lam ~ normal(0, 1);
  g ~ normal(0, lam);
  mu ~ normal(0, 1);
  y0 ~ normal(0, 1);

  for(t in 1:p)
    y[t] ~ normal(mu + y0[t], sigma);

  for(t in p + 1:T){
    real y_hat = 0;
    for(tau in 1:p)
      y_hat += b[tau] * y[t - tau];
    y[t] ~ normal(mu + y_hat, sigma);
  }
}

generated quantities {
  vector[T] y_ppc;

  for(t in 1:p)
    y_ppc[t] = normal_rng(mu + y0[t], sigma);

  for(t in p + 1:T){
    real y_hat = 0;
    for(tau in 1:p)
      y_hat += b[tau] * y[t - tau];
    y_ppc[t] = normal_rng(mu + y_hat, sigma);
  }
}

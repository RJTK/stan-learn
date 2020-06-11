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
  int<lower=1> N;  // Number of examples
  vector[N] y;  // the data series y(t)
}

transformed data {
}

parameters {
  real mu;  // Mean value
  real y0;  // Initial value
  vector<lower=-1, upper=1>[1] g;  // Reflection coefficients
  real<lower=0> sigma;  // noise level
  real<lower=0> lam;  // variance of reflection coefficients
}

transformed parameters {
  vector[1] b;  // AR Coefficients

  b = step_up_recursion(g);
}

model {
  sigma ~ normal(0, 1);
  lam ~ normal(0, 1);
  g ~ normal(0, lam);
  mu ~ normal(0, 1);
  y0 ~ normal(0, 1);

  y[1] ~ normal(mu + y0, sigma);
  y[2:N] ~ normal(mu + b[1] * y[1:N - 1], sigma);
}

generated quantities {
  vector[N] y_hat;

  y_hat[1] = normal_rng(mu + y0, sigma);
  for (t in 2:N)
    y_hat[t] = normal_rng(mu + b[1] * y[t - 1], sigma);
}

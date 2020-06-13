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

  vector reverse(vector x){
    // Isn't this built in?  stanc2.23 doesn't recognize it.
    int p = dims(x)[1];
    vector[p] x_rev;
    for (tau in 1:p)
      x_rev[tau] = x[p - tau + 1];
    return x_rev;
  }

  real ar_model_lpdf(vector y, vector y0, vector b, real mu, real sigma){
    int p = dims(b)[1];
    int T = dims(y)[1];
    real lpdf_ret = 0.0;
    vector[p] b_rev = reverse(b);

    for(t in 1:p)
      lpdf_ret += normal_lpdf(y[t] | mu + y0[t], sigma^2);

    for(t in p + 1:T){
      real y_hat = 0;
      y_hat = dot_product(b_rev, y[t - p:t - 1]);
      lpdf_ret += normal_lpdf(y[t] | mu + y_hat, sigma^2);
    }
    return lpdf_ret;
  }

  vector ar_model_rng(vector y, vector y0, vector b, real mu, real sigma){
    int p = dims(b)[1];
    int T = dims(y)[1];
    vector[T] y_1s; // 1 step ahead
    vector[p] b_rev = reverse(b);

    for(t in 1:p)
      y_1s[t] = normal_rng(mu + y0[t], sigma^2);

    for(t in p + 1:T){
      real y_hat = 0;
      y_hat = dot_product(b_rev, y[t - p:t - 1]);
      y_1s[t] = normal_rng(mu + y_hat, sigma^2);
    }
    return y_1s;
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
  vector[p] y0;  // Initial values
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
  y0 ~ normal(mu, sigma^2);
  y ~ ar_model(y0, b, mu, sigma);
}

generated quantities {
  vector[T] y_ppc;
  y_ppc = ar_model_rng(y, y0, b, mu, sigma);
}

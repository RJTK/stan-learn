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
    int p = dims(x)[1];
    vector[p] x_rev;
    for (tau in 1:p)
      x_rev[tau] = x[p - tau + 1];
    return x_rev;
  }

  vector ar_model_forecast(vector y, vector y0, vector b, real sigma){
    int p = dims(b)[1];
    int T = dims(y)[1];
    vector[T] y_hat;
    vector[p] b_rev = reverse(b);

    // Prepend the initial values y0
    vector[p + T] y_full;
    y_full[:p] = y0;
    y_full[p + 1:] = y;

    for(t in 1:T){
      y_hat[t] = dot_product(b_rev, y_full[t:t + p - 1]);
    }
    return y_hat;
  }

  real ar_model_lpdf(vector y, vector y0, vector b, real sigma){
    int T = dims(y)[1];
    vector[T] y_hat = ar_model_forecast(y, y0, b, sigma);
    return normal_lpdf(y | y_hat, sigma^2);
  }

  vector ar_model_rng(vector y, vector y0, vector b, real sigma){
    int T = dims(y)[1];
    vector[T] y_hat = ar_model_forecast(y, y0, b, sigma);
    vector[T] y_rng;
    for(t in 1:T)
      y_rng[t] = normal_rng(y_hat[t], sigma^2);
    return y_rng;
  }
}

data {
  int<lower=1> T;  // Number of examples
  int<lower=1> p;  // The lag
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
  real<lower=0> lam;  // magnitude of r
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
  vector[p + T] trend = mu + r * t;

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

  // trend parameters
  lam ~ normal(0, 1);
  r ~ normal(0, lam^2);
  mu ~ normal(0, 1);  // A mean offset

  y0 ~ normal(trend[:p], sigma^2);
  y - trend[p + 1:] ~ ar_model(y0, b, sigma);
}

generated quantities {
  vector[T] y_ppc;
  y_ppc = trend[p + 1:] + ar_model_rng(y, y0, b, sigma);
}
